"""SeaKR method-level reproduction (Self-aware Knowledge Retrieval).

SeaKR (Yao et al., 2024) drives retrieval with SELF-AWARE UNCERTAINTY computed
from the LLM's internal states across k sampled decodings, and uses the same
signal to RE-RANK retrieved knowledge (keep the passage that reduces
uncertainty most). Reproduced onto this stack at the method level, same
standard as probingrag/: identical backbone (Qwen3.5-9B), retriever
(E5/wiki18), and the project-wide direct-answer prompt.

Per item:
  1. closed-book: k sampled answers (vLLM, one request) -> HF forward per
     answer -> last-layer mean over ANSWER tokens -> k embeddings ->
     u = logdet(Gram(L2-normalized) + eps*I)  (spread of the k states;
     lower = samples agree = model "knows")
  2. u <= theta  -> stop, return the greedy answer (no retrieval)
  3. else retrieve k_docs passages; for EACH passage compute the same
     uncertainty conditioned on it; self-aware re-rank: keep argmin-u passage
  4. final answer = greedy generation with the chosen passage

Algorithm-level 1:1 with the official repo (THU-KEG/SeaKR), on OUR shared
Qwen3.5-9B + E5/wiki18 backbone (all systems must share one backbone/retriever
for a fair comparison — the same reason FlashRAG re-implements methods). Every
algorithmic choice matches the official source:
  - EigenScore uncertainty = log-det of the CENTERED covariance of the k
    per-sample hidden states (INSIDE);
  - states read at the LAST (EOS) token of each generation (official choice —
    we append the EOS id post-hoc and read there);
  - iterative multi-hop self-aware reasoning: max_reasoning_steps=7, retrieve
    when uncertain, SELF-AWARE RE-RANK the max_docs=5 candidates by the
    uncertainty of (context + candidate) and keep the argmin, accumulate,
    reason again until confident (u <= theta) or steps exhausted;
  - eigen_threshold tuned on dev (official ships ~ -6).
The ONE non-algorithmic difference (user-approved 2026-07-09): we extract the
EOS hidden state via a post-hoc HF forward over (prompt + sampled answer + EOS),
whereas the official fork reads it DURING generation from a modified vLLM. Same
tensor in principle, pure efficiency detail — it does not change the algorithm
or the ranking, only makes our SeaKR more expensive (extra forwards). k_samples
= 5 (user-set). Backbone-imposed: numbers cannot equal the paper's LLaMA-2
results; the goal is their exact algorithm on our Qwen backbone.

theta is tuned on dev by replaying recorded u values (collect with
record_u=True, sweep offline — no rerun needed).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ragprobe.features.extractor import _apply_chat_template

from pgir.gen_prompt import build_base_prompt
from pgir.strategies import render_contexts
from pgir.vllm_client import VLLMClient

EPS = 1e-4


class SeaKR:
    def __init__(self, model_name: str, vllm_url: str, retriever,
                 theta: float = 0.0, k_samples: int = 5, k_docs: int = 5,
                 max_rounds: int = 7, temperature: float = 1.0,
                 device: str = "cuda", record_u: bool = False):
        # defaults aligned to the official repo: max_reasoning_steps=7,
        # max_docs=5, EigenScore threshold ~ -6; k_samples=5 (user-set, cost).
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"
        self.llm = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=device).eval()
        self.vllm = VLLMClient(vllm_url, model_name)
        self.retriever = retriever
        self.theta = float(theta)
        self.k_samples = int(k_samples)
        self.k_docs = int(k_docs)
        self.max_rounds = int(max_rounds)
        self.temperature = float(temperature)
        self.record_u = record_u  # tuning mode: never stop early, log all u
        # create the cublas handle NOW — after the 64GB faiss index loads,
        # late cublasCreate fails with CUBLAS_STATUS_ALLOC_FAILED (pinned-mem
        # fragmentation); construct this runner BEFORE the retriever.
        with torch.no_grad():
            ids = self.processor.tokenizer("warmup", return_tensors="pt"
                                           ).input_ids.to(self.llm.device)
            self.llm(input_ids=ids, use_cache=False)

    # ------------------------------------------------------------ internals
    @torch.no_grad()
    def _answer_embeddings(self, question: str, contexts, answers: list[str],
                           batch_size: int = 8) -> np.ndarray:
        """Last-layer hidden state at the trailing EOS token, one row per answer.

        Official SeaKR extracts the EigenScore states at the LAST (EOS) token of
        each sampled generation — the position where the model decides to stop.
        Post-hoc we don't see the model's own EOS, so we APPEND the tokenizer
        EOS id to (prompt + answer) and read the hidden state at that appended
        EOS position (right padding keeps it at index len-1 of the unpadded ids)."""
        tok = self.processor.tokenizer
        eos_id = tok.eos_token_id
        vecs = []
        for s in range(0, len(answers), batch_size):
            chunk = answers[s:s + batch_size]
            id_lists, eos_pos = [], []
            for ans in chunk:
                _, base_prompt = build_base_prompt(question, contexts or None)
                combined = base_prompt + " " + ans
                enc = _apply_chat_template(self.processor, combined, "cpu")
                ids = enc["input_ids"][0].tolist() + [eos_id]  # append EOS
                id_lists.append(ids)
                eos_pos.append(len(ids) - 1)                   # the EOS position
            batch = tok.pad({"input_ids": id_lists}, padding=True,
                            return_tensors="pt").to(self.llm.device)
            out = self.llm(**batch, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[-1]
            for j, p in enumerate(eos_pos):
                vecs.append(h[j, p].float().cpu().numpy())
            del out
        return np.stack(vecs)

    def _uncertainty(self, question: str, contexts, answers: list[str]) -> float:
        """EigenScore: log-det of the (centered) covariance Gram of the k
        last-token embeddings (INSIDE / SeaKR). Lower = samples agree = the
        model is confident. Any monotone transform is threshold-tuned on dev."""
        answers = [a for a in answers if a.strip()]
        if len(answers) < 2:
            return float("inf")  # degenerate sampling = maximally uncertain
        v = self._answer_embeddings(question, contexts, answers)
        v = v - v.mean(axis=0, keepdims=True)          # center across samples
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        g = v @ v.T                                    # k x k covariance Gram
        sign, logdet = np.linalg.slogdet(g + EPS * np.eye(len(v)))
        return float(logdet)

    def _sample_answers(self, question: str, contexts) -> list[str]:
        return self.vllm.generate_samples(question, contexts,
                                          n=self.k_samples,
                                          temperature=self.temperature)

    def _greedy_answer(self, question: str, contexts) -> str:
        return self.vllm.generate(question, contexts)

    # ------------------------------------------------------------ main loop
    def run_items(self, items: list[dict]) -> list[dict]:
        """Iterative self-aware reasoning (official SeaKR multi-hop):

        Repeat up to max_rounds. Each round measures EigenScore uncertainty on
        the CURRENT accumulated context; if confident (u <= theta) stop and
        return that round's greedy answer; else retrieve (answer-guided query,
        used pids excluded), SELF-AWARE RE-RANK the candidates by the
        uncertainty of context+candidate, and ADD the argmin-u passage to the
        running context, then reason again.

        record_u mode never stops early: it runs the full max_rounds and stores
        per-round (u, greedy answer) so any theta is replayable offline — stop
        at the first round whose u <= theta, else the last round's answer."""
        traces = []
        for it in items:
            t0 = time.monotonic()
            q = it["question"]
            gen_calls = retr_calls = hf_forwards = 0
            context_docs: list[dict] = []
            used_ids: set = set()
            u_trace: list[float] = []       # per-round uncertainty (current context)
            ans_trace: list[str] = []       # per-round greedy answer at that state
            picked: list[str] = []          # pids added each round
            final_answer = None
            stop_round = None

            for step in range(self.max_rounds):
                ctx = render_contexts(context_docs) if context_docs else None
                samples = self._sample_answers(q, ctx)
                gen_calls += self.k_samples
                hf_forwards += self.k_samples
                u = self._uncertainty(q, ctx, samples)
                greedy = self._greedy_answer(q, ctx)
                gen_calls += 1
                u_trace.append(u)
                ans_trace.append(greedy)

                if u <= self.theta and not self.record_u:
                    final_answer, stop_round = greedy, step
                    break
                if step == self.max_rounds - 1:
                    final_answer, stop_round = greedy, step
                    break

                # retrieve more (answer-guided, exclude used) + self-aware rerank
                query = (q + " " + greedy).strip()
                cand = [d for d in self.retriever.search(query, self.k_docs + len(used_ids))
                        if d.get("id") not in used_ids][:self.k_docs]
                retr_calls += 1
                if not cand:
                    final_answer, stop_round = greedy, step
                    break
                per_doc_u = []
                for d in cand:
                    cctx = render_contexts(context_docs + [d])
                    s = self._sample_answers(q, cctx)
                    gen_calls += self.k_samples
                    hf_forwards += self.k_samples
                    per_doc_u.append(self._uncertainty(q, cctx, s))
                best = int(np.argmin(per_doc_u))
                context_docs.append(cand[best])
                used_ids.add(cand[best].get("id"))
                picked.append(cand[best].get("id"))

            rec = {"example_id": it["example_id"], "question": q,
                   "answer": final_answer if final_answer is not None else ans_trace[-1],
                   "retrieved": bool(picked),
                   "u_trace": u_trace, "ans_trace": ans_trace,
                   "picked_docs": picked, "stop_round": stop_round,
                   # back-compat fields (first round == closed-book)
                   "u_closed": u_trace[0] if u_trace else float("inf"),
                   "cost": {"gen_calls": gen_calls, "retr_calls": retr_calls,
                            "hf_forwards": hf_forwards},
                   "wall_s": round(time.monotonic() - t0, 2)}
            traces.append(rec)
        return traces


def replay_theta(traces: list[dict], theta: float) -> dict[str, str]:
    """Offline theta sweep on record_u traces: stop at the first round whose
    uncertainty <= theta (return that round's greedy answer), else the last."""
    out = {}
    for tr in traces:
        us = tr.get("u_trace") or [tr.get("u_closed", float("inf"))]
        ans = tr.get("ans_trace") or [tr.get("answer", "")]
        idx = next((i for i, u in enumerate(us) if u <= theta), len(us) - 1)
        out[tr["example_id"]] = ans[idx] if ans else ""
    return out
