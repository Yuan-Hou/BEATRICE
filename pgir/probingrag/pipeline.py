"""Probing-RAG inference pipeline (method reproduction, see package docstring).

Loop per item (batched across items per round):
  round 0: closed-book answer (vLLM, training-format prompt)
  probe:   HF forward over (prompt + answer), mean-pool ANSWER-token hidden
           states at the selected layers -> probers -> summed logit margin
           (margin > theta  =>  "no retrieval needed" => stop with this answer)
  round t: retrieve (q at t=1; q + last answer after, deduped), regenerate, probe
  stop at max_iter regardless; final answer = last generation (NO pool argmax —
  that is the key contrast with PGIR).

Feature parity with prober training: prompt built by pgir.gen_prompt (service
code), answer-span located by the extractor's own `_find_text_span`, mean over
answer tokens — identical to scripts/extract_layer_features.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ragprobe.features.extractor import _apply_chat_template, _find_text_span

from pgir.gen_prompt import build_base_prompt
from pgir.strategies import _doc_id, render_contexts
from pgir.vllm_client import VLLMClient

import os as _os

ASSETS = Path(_os.environ.get(
    "PGIR_PROBINGRAG_PROBERS",
    Path(__file__).resolve().parent.parent / "assets" / "probingrag_probers"))


class ProberBank:
    def __init__(self, device: str = "cuda"):
        from pgir.probingrag.train_probers import Prober

        cfg = json.loads((ASSETS / "config.json").read_text())
        self.layers = cfg["layers"]
        self.device = device
        self.probers = {}
        for l in self.layers:
            p = Prober(cfg["dim"], cfg["hidden"]).to(device).eval()
            p.load_state_dict(torch.load(ASSETS / f"prober_layer{l}.pt", map_location=device))
            self.probers[l] = p

    @torch.no_grad()
    def margin(self, pooled: dict[int, torch.Tensor]) -> float:
        """pooled: layer -> (4096,) fp32 answer-token mean. Returns summed margin."""
        total = 0.0
        for l in self.layers:
            logits = self.probers[l](pooled[l].unsqueeze(0).to(self.device))[0]
            total += float(logits[1] - logits[0])  # >0: no retrieval needed
        return total


class ProbingRAG:
    def __init__(self, model_name: str, vllm_url: str, retriever,
                 theta: float = 0.0, max_iter: int = 4, k: int = 3,
                 device: str = "cuda", record_all_rounds: bool = False):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.bank = ProberBank(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        # spans are computed on unpadded ids -> padding MUST be on the right
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"
        self.llm = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=device).eval()
        self.device = device
        self.vllm = VLLMClient(vllm_url, model_name)
        self.retriever = retriever
        self.theta = float(theta)
        self.max_iter = int(max_iter)
        self.k = int(k)
        self.record_all_rounds = record_all_rounds  # theta-tuning mode: never stop early

    @torch.no_grad()
    def _probe_margin(self, question: str, contexts, answer: str) -> float:
        return self._probe_margins([(question, contexts, answer)])[0]

    @torch.no_grad()
    def _probe_margins(self, triples: list[tuple], batch_size: int = 8) -> list[float]:
        """Batched probing forwards (right padding; spans on unpadded ids)."""
        tok = self.processor.tokenizer
        out_margins: list[float] = [0.0] * len(triples)
        order = sorted(range(len(triples)),
                       key=lambda i: len(str(triples[i][1] or "")) )
        for s in range(0, len(order), batch_size):
            chunk = order[s:s + batch_size]
            id_lists, spans = [], []
            for i in chunk:
                question, contexts, answer = triples[i]
                _, base_prompt = build_base_prompt(question, contexts or None)
                combined = base_prompt + " " + answer
                enc = _apply_chat_template(self.processor, combined, "cpu")
                ids = enc["input_ids"][0]
                span = _find_text_span(tok, ids, answer,
                                       min_start=max(0, len(ids) - 200))
                if span is None:
                    span = (max(0, len(ids) - 32), len(ids))
                id_lists.append(ids.tolist())
                spans.append(span)
            batch = tok.pad({"input_ids": id_lists}, padding=True,
                            return_tensors="pt").to(self.llm.device)
            out = self.llm(**batch, output_hidden_states=True, use_cache=False)
            for j, i in enumerate(chunk):
                a, b = spans[j]
                pooled = {l: out.hidden_states[l][j, a:b].mean(0).float().cpu()
                          for l in self.bank.layers}
                out_margins[i] = self.bank.margin(pooled)
            del out
        return out_margins

    def run_items(self, items: list[dict]) -> list[dict]:
        """items: [{example_id, question}]. Returns per-item trace dicts."""
        state = {i: {"docs": None, "used": set(), "rounds": [],
                     "final": None, "stop_round": None}
                 for i in range(len(items))}
        active = list(range(len(items)))
        gen_calls = [0] * len(items)
        retr_calls = [0] * len(items)

        for t in range(self.max_iter + 1):
            if not active:
                break
            # 1) batched generation via vLLM
            qc = [(items[i]["question"],
                   render_contexts(state[i]["docs"]) if state[i]["docs"] else None)
                  for i in active]
            answers = self.vllm.generate_many(qc)
            # 2) batched probing forwards
            triples = [(items[i]["question"],
                        render_contexts(state[i]["docs"]) if state[i]["docs"] else None,
                        ans) for i, ans in zip(active, answers)]
            margins = self._probe_margins(triples)
            still = []
            for (i, ans), m in zip(zip(active, answers), margins):
                gen_calls[i] += 1
                state[i]["rounds"].append({"round": t, "answer": ans, "margin": m,
                                           "pids": [ _doc_id(d) for d in (state[i]["docs"] or [])]})
                stop = (m > self.theta) and not self.record_all_rounds
                if stop or t == self.max_iter:
                    if state[i]["final"] is None:
                        state[i]["final"] = ans
                        state[i]["stop_round"] = t
                    if t == self.max_iter and state[i]["final"] is None:
                        state[i]["final"] = ans
                else:
                    still.append(i)
            active = still
            # 3) batched retrieval for the survivors
            if active:
                queries = []
                for i in active:
                    q = items[i]["question"]
                    if state[i]["docs"] is not None:
                        last = state[i]["rounds"][-1]["answer"]
                        q = (q + " " + " ".join(str(last).split()[:32])).strip()
                    queries.append(q)
                overfetch = self.k + 4 * (t + 1)
                batched = self.retriever.batch_search(queries, num=overfetch)
                for i, fetched in zip(active, batched):
                    retr_calls[i] += 1
                    fresh = [d for d in fetched if _doc_id(d) not in state[i]["used"]]
                    docs = fresh[:self.k] or fetched[:self.k]
                    state[i]["used"] |= {_doc_id(d) for d in docs}
                    state[i]["docs"] = docs

        out = []
        for i, item in enumerate(items):
            s = state[i]
            if s["final"] is None and s["rounds"]:
                s["final"] = s["rounds"][-1]["answer"]
                s["stop_round"] = s["rounds"][-1]["round"]
            out.append({"example_id": item.get("example_id"), "question": item["question"],
                        "answer": s["final"] or "", "stop_round": s["stop_round"],
                        "rounds": s["rounds"],
                        "cost": {"gen_calls": gen_calls[i], "retr_calls": retr_calls[i]}})
        return out


def replay_theta(traces: list[dict], theta: float, max_iter: int = 4) -> dict[str, str]:
    """Offline theta replay over record_all_rounds traces: id -> final answer."""
    final = {}
    for tr in traces:
        ans = tr["rounds"][-1]["answer"] if tr["rounds"] else ""
        for r in tr["rounds"]:
            if r["margin"] > theta or r["round"] == max_iter:
                ans = r["answer"]
                break
        final[tr["example_id"]] = ans
    return final
