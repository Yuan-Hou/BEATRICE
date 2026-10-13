"""evaluate_arm: one call = evaluate one (question, evidence) arm (spec §3).

Backends:
  SCORER — one /score call WITHOUT `answer`; the service generates with the
           training prompt and scores atomically. Reference implementation,
           slow (eager). Development / smoke default.
  VLLM   — vLLM generates the answer (training-identical prompt via
           gen_prompt), then /score WITH `answer`. Production default.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .scorer_client import ScorerClient


@dataclass
class Cost:
    gen_calls: int = 0
    score_calls: int = 0
    retr_calls: int = 0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            self.gen_calls + other.gen_calls,
            self.score_calls + other.score_calls,
            self.retr_calls + other.retr_calls,
        )

    def to_dict(self) -> dict:
        return {"gen_calls": self.gen_calls, "score_calls": self.score_calls,
                "retr_calls": self.retr_calls}


@dataclass
class Arm:
    answer: str
    p_raw: float
    p_cal: float
    mode: str                      # "question" (closed-book) | "context"
    branch_probs: dict = field(default_factory=dict)
    answer_source: str = ""        # provided | generated
    probe_checkpoint: str = ""
    cost: Cost = field(default_factory=Cost)
    round_index: int = -1
    evidence_pids: list = field(default_factory=list)
    contexts: list = field(default_factory=list)   # [{"text": ...}] as scored
    status: str = "success"
    error: str | None = None
    # per-call wall times (serial-latency accounting): gen_s = vLLM generation
    # (0 for SCORER backend — included in score), score_s = scorer client wall,
    # scorer_server_s = scorer-reported forward time
    timing: dict = field(default_factory=dict)
    # dual q+ctx fusion diagnostics (two-phase service flow): fusion kind,
    # p_context_only / p_qonly components, qonly_uuid. Empty on old traces.
    fusion: dict = field(default_factory=dict)

    @property
    def serial_s(self) -> float:
        return float(self.timing.get("gen_s", 0.0)) + float(self.timing.get("score_s", 0.0))

    def to_dict(self) -> dict:
        return {
            "answer": self.answer, "p_raw": self.p_raw, "p_cal": self.p_cal,
            "mode": self.mode, "branch_probs": self.branch_probs,
            "answer_source": self.answer_source, "probe_checkpoint": self.probe_checkpoint,
            "cost": self.cost.to_dict(), "round_index": self.round_index,
            "evidence_pids": list(self.evidence_pids), "contexts": list(self.contexts),
            "status": self.status, "error": self.error, "timing": dict(self.timing),
            "fusion": dict(self.fusion),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Arm":
        d = dict(d)
        d["cost"] = Cost(**d.get("cost", {}))
        return cls(**d)


class ArmEvaluator:
    def __init__(
        self,
        scorer: ScorerClient,
        backend: str = "vllm",           # "vllm" | "scorer"
        vllm=None,                        # VLLMClient, required for backend="vllm"
        calibrator=None,                  # calibration.PerModeCalibrator | None
    ):
        backend = backend.lower()
        if backend not in ("vllm", "scorer"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "vllm" and vllm is None:
            raise ValueError("backend='vllm' requires a VLLMClient")
        self.scorer = scorer
        self.backend = backend
        self.vllm = vllm
        self.calibrator = calibrator
        # example -> qonly_uuid from /score_question (round-0 greedy closed-book);
        # later /score_context calls of the same example reuse it for dual fusion.
        # Tiny (one uuid string per item this process has seen); GIL-safe dict ops.
        self._qonly_uuids: dict[str, str] = {}

    @staticmethod
    def _uuid_key(job: dict[str, Any]) -> str:
        return str(job.get("example_id") or job.get("question") or "")

    # one arm (convenience) -------------------------------------------------
    def evaluate(self, question: str, contexts=None, example_id=None,
                 evidence_pids=None, round_index: int = -1) -> Arm:
        return asyncio.run(self.aevaluate_many(
            [dict(question=question, contexts=contexts, example_id=example_id,
                  evidence_pids=evidence_pids, round_index=round_index)]))[0]

    # many arms (trace collection / batched rounds) -------------------------
    async def aevaluate_many(self, jobs: list[dict[str, Any]]) -> list[Arm]:
        """Jobs may carry a pre-generated "answer" (best-of-N sampled candidates):
        those skip generation and go straight to scoring. Their 1 gen call was
        already paid inside the sampling request, so per-arm cost stays 1 gen +
        1 score either way — uniform accounting."""
        answers: list[str | None] = [j.get("answer") or None for j in jobs]
        # The two-phase flow (/score_question caches q-only vecs -> /score_context
        # fuses them) only buys anything when the service serves a dual q+ctx head.
        # Without one (joint fusion dropped 2026-07-16) it is pure overhead: those
        # endpoints bypass the server's micro-batcher. Fall back to plain /score.
        two_phase = bool(getattr(self.scorer, "dual_fusion", None))
        # greedy closed-book jobs (no contexts, no pre-provided answer) are the
        # canonical q-only arms -> /score_question (caches q branch vecs, returns
        # qonly_uuid). Sampled q-mode siblings keep the plain /score path.
        qonly_greedy = [two_phase and not j.get("contexts") and a is None
                        for j, a in zip(jobs, answers)]
        gen_times: list[float] = [0.0] * len(jobs)
        gen_cost = 0
        need_gen = [i for i, a in enumerate(answers) if a is None]
        if self.backend == "vllm" and need_gen:
            timed = await self.vllm.agenerate_many_timed(
                [(jobs[i]["question"], jobs[i].get("contexts")) for i in need_gen])
            for i, (text, secs) in zip(need_gen, timed):
                answers[i] = text
                gen_times[i] = secs
            gen_cost = 1
        elif self.backend == "vllm":
            gen_cost = 1  # all answers provided; their generation was the sampling call
        items = []
        for j, ans, is_q0 in zip(jobs, answers, qonly_greedy):
            item = {"question": j["question"], "example_id": j.get("example_id")}
            if j.get("contexts"):
                item["contexts"] = j["contexts"]
                if two_phase:
                    item["_endpoint"] = "score_context"
                    uid = self._qonly_uuids.get(self._uuid_key(j))
                    if uid:
                        item["qonly_uuid"] = uid
            elif two_phase:
                # ALL question-mode arms (greedy + bon-sampled) go through
                # /score_question so scorer traffic stays on one path, covered
                # by the client-side in-flight cap. Only greedy q0 uuids are kept.
                item["_endpoint"] = "score_question"
            if ans:
                item["answer"] = ans
            items.append(item)
        results = await self.scorer.ascore_many(items)
        for j, is_q0, res in zip(jobs, qonly_greedy, results):
            if is_q0 and res.get("status") == "success" and res.get("qonly_uuid"):
                self._qonly_uuids[self._uuid_key(j)] = res["qonly_uuid"]
        extra_cost = await self._heal_expired_qonly(jobs, items, results)

        arms = []
        for idx, (j, item, res, gen_s) in enumerate(zip(jobs, items, results, gen_times)):
            mode = "context" if j.get("contexts") else "question"
            timing = {"gen_s": round(gen_s, 4),
                      "score_s": res.get("client_elapsed_s", 0.0),
                      "scorer_server_s": res.get("timing_s", 0.0)}
            # the SCORER backend generates inside the service => still 1 gen call
            cost = Cost(gen_calls=max(gen_cost, 1), score_calls=1, retr_calls=0)
            if idx in extra_cost:  # qonly-cache refresh surcharge (honest accounting)
                cost = cost + extra_cost[idx]
            if res.get("status") != "success":
                arms.append(Arm(answer="", p_raw=0.0, p_cal=0.0, mode=mode, cost=cost,
                                round_index=j.get("round_index", -1),
                                evidence_pids=j.get("evidence_pids") or [],
                                contexts=item.get("contexts") or [],
                                status="failed", error=res.get("error"), timing=timing))
                continue
            p_raw = float(res["p_correct"])
            p_cal = (self.calibrator.calibrate(p_raw, mode)
                     if self.calibrator is not None else p_raw)
            fus = {}
            if res.get("fusion"):
                fus["fusion"] = res["fusion"]
            for src, dst in (("p_correct_context_only", "p_context_only"),
                             ("p_correct_qonly", "p_qonly"),
                             ("qonly_uuid", "qonly_uuid"),
                             ("fusion_refreshed", "refreshed")):
                if res.get(src) is not None:
                    fus[dst] = res[src]
            arms.append(Arm(
                answer=res.get("answer", ""), p_raw=p_raw, p_cal=p_cal, mode=mode,
                branch_probs=res.get("branch_probs") or {},
                answer_source=j.get("answer_source") or res.get("answer_source", ""),
                probe_checkpoint=res.get("probe_checkpoint", ""),
                cost=cost, round_index=j.get("round_index", -1),
                evidence_pids=j.get("evidence_pids") or [],
                contexts=item.get("contexts") or [],
                timing=timing, fusion=fus,
            ))
        return arms

    async def _heal_expired_qonly(self, jobs, items, results) -> dict[int, Cost]:
        """The service's qonly cache is bounded (LRU + TTL ~30min): a
        /score_context call can come back degraded to context_only if the uuid
        expired mid-item. Refresh phase 1 ONCE (the service regenerates the
        greedy closed-book answer -> same branch vecs) and re-score those ctx
        items fused. Returns per-index surcharge Costs (results patched in place)."""
        dual = getattr(self.scorer, "dual_fusion", None)
        stale = [i for i, (it, res) in enumerate(zip(items, results))
                 if it.get("_endpoint") == "score_context"
                 and res.get("status") == "success"
                 and str(res.get("fusion", "")).startswith("context_only")]
        if not stale or not dual:
            return {}
        by_key: dict[str, list[int]] = {}
        for i in stale:
            by_key.setdefault(self._uuid_key(jobs[i]), []).append(i)
        q_items = [{"question": jobs[idxs[0]]["question"],
                    "example_id": jobs[idxs[0]].get("example_id"),
                    "_endpoint": "score_question"} for idxs in by_key.values()]
        q_results = await self.scorer.ascore_many(q_items)
        extra: dict[int, Cost] = {}
        retry_items, retry_pos = [], []
        for (key, idxs), qres in zip(by_key.items(), q_results):
            if qres.get("status") != "success" or not qres.get("qonly_uuid"):
                continue  # keep the context_only result; fusion field discloses it
            self._qonly_uuids[key] = qres["qonly_uuid"]
            # phase-1 refresh = 1 gen (in-service greedy) + 1 score, charged once
            extra[idxs[0]] = Cost(gen_calls=1, score_calls=1)
            for i in idxs:
                it = dict(items[i])
                it["qonly_uuid"] = qres["qonly_uuid"]
                retry_items.append(it)
                retry_pos.append(i)
        if retry_items:
            retry_results = await self.scorer.ascore_many(retry_items)
            for i, res in zip(retry_pos, retry_results):
                if res.get("status") == "success":
                    res["fusion_refreshed"] = True
                    results[i] = res
                    extra[i] = extra.get(i, Cost()) + Cost(score_calls=1)  # the re-score
        return extra

    def evaluate_many(self, jobs: list[dict[str, Any]]) -> list[Arm]:
        return asyncio.run(self.aevaluate_many(jobs))
