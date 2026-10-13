"""DRAGIN method-level reproduction (RIND trigger + QFS query formulation).

DRAGIN (Su et al., ACL 2024) decides WHEN to retrieve from the model's own
generation dynamics — a token triggers retrieval when it is uncertain (high
entropy), influential (receives attention from later tokens) and semantic
(not a stopword) — and WHAT to retrieve from attention (top attended tokens
form the query). Reproduced onto this stack at the method level, same
standard as probingrag/ and seakr.py.

Per item, up to max_rounds:
  1. reasoning generation (vLLM TRUE continuation: chat template ids +
     accumulated reasoning prefix ids -> /v1/completions), 64 tokens/step;
  2. ONE HF forward over (prompt + reasoning) with eager attention ->
     for each NEW token i: RIND_i = H_i * A_i * s_i where
       H_i = entropy of the predictive distribution that emitted token i,
       A_i = max attention token i receives from later tokens
             (last usable attention layer, max over heads),
       s_i = 0 if stopword/punctuation else 1;
  3. no token above theta -> keep reasoning, stop the loop;
     else truncate at the FIRST triggering token, form the QFS query
     (attention row of the trigger token over the user message, top-n
     non-stopword tokens in original order, prefixed with the question),
     retrieve k docs (replace evidence), continue from the truncated prefix;
  4. final answer = the project-wide direct-answer prompt over the final
     evidence set (EM-parity adaptation, same answer format as every system
     in the table; documented).

Simplifications vs official repo: RIND computed post-hoc in one forward
(identical quantities, not token-streamed); QFS query keeps the original
question as prefix; final short answer via the unified prompt. Qwen3.5 is a
hybrid-attention model — only some layers expose full attention maps; we use
the LAST layer that returns a usable square attention matrix.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ragprobe.features.extractor import _apply_chat_template

from pgir.gen_prompt import build_base_prompt
from pgir.strategies import render_contexts
from pgir.vllm_client import VLLMClient

STOPWORDS = set("""a an the and or but if then else of in on at to from by for
with about into over after under again further is are was were be been being
do does did have has had this that these those it its as not no nor so than
too very can will just don should now he she they them his her their we you
your i me my what which who whom when where why how all any both each few
more most other some such only own same s t d ll m o re ve y ain aren""".split())

REASON_INSTRUCTION = (
    "Answer the question by reasoning step by step. Keep the reasoning short "
    "and end with \"So the answer is <answer>.\"")


def _is_stop(tok_text: str) -> bool:
    t = tok_text.strip().lower()
    return (not t) or (not any(c.isalnum() for c in t)) or t in STOPWORDS


class DRAGIN:
    def __init__(self, model_name: str, vllm_url: str, retriever,
                 theta: float = 1.0, max_rounds: int = 2, k_docs: int = 3,
                 step_tokens: int = 64, qfs_top: int = 8,
                 device: str = "cuda", record_signals: bool = False):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name)
        tok = self.processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # eager attention: sdpa does not return attention weights
        self.llm = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map=device).eval()
        self.vllm = VLLMClient(vllm_url, model_name)
        self.retriever = retriever
        self.theta = float(theta)
        self.max_rounds = int(max_rounds)
        self.k_docs = int(k_docs)
        self.step_tokens = int(step_tokens)
        self.qfs_top = int(qfs_top)
        self.record_signals = record_signals
        # early cublas handle creation — see seakr.py note (must precede the
        # 64GB faiss index load or cublasCreate can fail with ALLOC_FAILED)
        with torch.no_grad():
            ids = tok("warmup", return_tensors="pt").input_ids.to(self.llm.device)
            self.llm(input_ids=ids, use_cache=False)

    # ------------------------------------------------------------ internals
    def _user_msg(self, question: str, docs) -> str:
        parts = []
        if docs:
            ctx = "\n".join(f"[{i+1}] {d.get('contents', '')}"
                            for i, d in enumerate(docs))
            parts.append(f"Context:\n{ctx}\n")
        parts.append(f"{REASON_INSTRUCTION}\nQuestion: {question}")
        return "\n".join(parts)

    @torch.no_grad()
    def _rind(self, user_text: str, reasoning: str) -> dict:
        """One eager forward; RIND + QFS materials for the reasoning tokens."""
        tok = self.processor.tokenizer
        base = _apply_chat_template(self.processor, user_text, "cpu")["input_ids"][0]
        gen_ids = tok.encode(reasoning, add_special_tokens=False)
        if not gen_ids:
            return {"rind": [], "tokens": [], "gen_start": len(base)}
        ids = torch.cat([base, torch.tensor(gen_ids)]).unsqueeze(0).to(self.llm.device)
        out = self.llm(input_ids=ids, output_attentions=True, use_cache=False)
        # last usable full-attention layer. Qwen3.5 hybrid: some layers give
        # None -> take the last square one. Gemma-E hybrid: EVERY layer returns
        # a square map but most are sliding-window (QFS needs long-range rows),
        # so when the config declares layer_types, restrict to full_attention.
        layer_types = getattr(self.llm.config, "layer_types", None) or getattr(
            getattr(self.llm.config, "text_config", None), "layer_types", None)
        cand = out.attentions
        if layer_types is not None and len(layer_types) == len(cand):
            cand = [a for a, t in zip(cand, layer_types) if t == "full_attention"]
        attn = None
        for a in reversed(cand):
            if a is not None and a.dim() == 4 and a.shape[-1] == a.shape[-2]:
                attn = a[0].float()  # (heads, L, L)
                break
        L = ids.shape[1]
        g0 = len(base)  # first generated-token position
        logits = out.logits[0].float()
        rind, toks = [], []
        for pos in range(g0, L):
            probs = torch.softmax(logits[pos - 1], dim=-1)
            h = float(-(probs * torch.log(probs + 1e-10)).sum())
            if attn is not None and pos + 1 < L:
                a_recv = float(attn[:, pos + 1:, pos].max())
            else:
                a_recv = 1.0  # last token / no attention available
            text = tok.decode([int(ids[0, pos])])
            s = 0.0 if _is_stop(text) else 1.0
            rind.append(h * a_recv * s)
            toks.append(text)
        # QFS material: trigger token's attention over the user message span
        qfs_rows = attn.mean(0) if attn is not None else None  # (L, L)
        del out
        return {"rind": rind, "tokens": toks, "gen_start": g0,
                "qfs": qfs_rows, "ids": ids[0].cpu()}

    def _qfs_query(self, question: str, sig: dict, trig_idx: int) -> str:
        """Top attended non-stopword user-span tokens, original order."""
        if sig["qfs"] is None:
            return question
        tok = self.processor.tokenizer
        pos = sig["gen_start"] + trig_idx
        row = sig["qfs"][pos, :sig["gen_start"]]
        order = torch.argsort(row, descending=True).tolist()
        picked = []
        for p in order:
            text = tok.decode([int(sig["ids"][p])])
            if _is_stop(text) or text.strip() in picked:
                continue
            picked.append((p, text.strip()))
            if len(picked) >= self.qfs_top:
                break
        picked.sort(key=lambda x: x[0])
        extra = " ".join(t for _, t in picked)
        return f"{question} {extra}".strip()

    # ------------------------------------------------------------ main loop
    def run_items(self, items: list[dict]) -> list[dict]:
        traces = []
        for it in items:
            t0 = time.monotonic()
            q = it["question"]
            docs, prefix = [], ""
            gen_calls = retr_calls = hf_forwards = 0
            rounds_log = []
            for r in range(self.max_rounds + 1):
                user = self._user_msg(q, docs)
                step = self.vllm.generate_continuation(
                    user, prefix, max_tokens=self.step_tokens)
                gen_calls += 1
                sig = self._rind(user, (prefix + " " + step).strip()
                                 if prefix else step)
                hf_forwards += 1
                # only NEW tokens may trigger (prefix was already inspected)
                tok = self.processor.tokenizer
                n_prefix = len(tok.encode(prefix, add_special_tokens=False)) if prefix else 0
                new_rind = sig["rind"][n_prefix:]
                trig = next((i for i, v in enumerate(new_rind)
                             if v > self.theta), None)
                rounds_log.append({
                    "round": r, "step_text": step,
                    "max_rind": max(new_rind) if new_rind else 0.0,
                    "triggered": trig is not None})
                if trig is None or r == self.max_rounds:
                    prefix = (prefix + " " + step).strip()
                    break
                # truncate at trigger, reformulate, retrieve, continue
                keep = tok.decode(tok.encode(step, add_special_tokens=False)[:trig])
                prefix = (prefix + " " + keep).strip()
                query = self._qfs_query(q, sig, n_prefix + trig)
                docs = self.retriever.search(query, self.k_docs)
                retr_calls += 1
                rounds_log[-1]["qfs_query"] = query
            # unified direct-answer generation over the final evidence (EM parity)
            contexts = render_contexts(docs) if docs else None
            answer = self.vllm.generate(q, contexts)
            gen_calls += 1
            tr = {"example_id": it["example_id"], "question": q,
                  "answer": answer, "reasoning": prefix,
                  "n_rounds": len(rounds_log),
                  "retrieved": bool(docs),
                  "cost": {"gen_calls": gen_calls, "retr_calls": retr_calls,
                           "hf_forwards": hf_forwards},
                  "wall_s": round(time.monotonic() - t0, 2)}
            if self.record_signals:
                tr["rounds"] = rounds_log
            traces.append(tr)
        return traces
