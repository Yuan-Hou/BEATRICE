"""PGIR control loop (spec §4).

`decide` is a pure function over the arm pool — unit-tested table-driven.
`PGIR` is framework-free (any retriever with .search); the FlashRAG
BasicPipeline wrapper lives in `flashrag_adapter.py` so this module imports
without FlashRAG installed.

Structural anti-regression guarantee: the closed-book arm enters the pool at
round 0 and the final answer is always the pool's p_cal-argmax (or an
early-accepted arm, which by construction is that argmax) — retrieval can
never make the final answer worse than the best already-evaluated arm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .arm import Arm, Cost
from .steer import REWRITE_MAX_TOKENS, build_rewrite_prompt, clean_rewrite, decide_steer
from .strategies import EvidenceState, next_evidence, render_contexts, s3_candidates


# --------------------------------------------------------------- policy ----
@dataclass(frozen=True)
class PolicyCfg:
    tau_hi: float = 0.8          # accept early if best p_cal >= tau_hi
    budget: int = 4              # max rounds (round 0 = closed-book, then B-1 evidence rounds)
    k: int = 3                   # passages per retrieval
    tau_lo: Optional[float] = None   # optional abstain threshold at budget (default off)
    # --- best-of-N answer sampling (SELECT; default off = current behavior) ---
    bon_n: int = 1               # candidates per round incl. the greedy one (1 = off)
    bon_lo: float = 0.05         # sample only when greedy p_cal in [bon_lo, tau_hi):
    #                              below = evidence hopeless, above = accepted anyway
    bon_temperature: float = 0.8
    # --- probe-steered evidence actions (STEER; default off) -----------------
    steer: bool = False
    steer_delta: float = 0.15    # ctx best below q0 best by delta => lock-in guard (s1)
    steer_eps: float = 0.02      # per-round ctx best stagnant within eps => rewrite query
    # --- efficiency knobs (M6 econ; default off = legacy behavior) -----------
    tau0: Optional[float] = None     # round-0 accept threshold (closed-book waiver)
    patience: Optional[int] = None   # stop after N stagnant rounds -> argmax
    patience_delta: float = 0.02


@dataclass(frozen=True)
class Accept:
    arm: Arm


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class ReturnArgmax:
    arm: Arm


@dataclass(frozen=True)
class Abstain:
    pass


Action = Any  # Accept | Continue | ReturnArgmax | Abstain


def _stagnant_rounds(pool: list[Arm], delta: float) -> int:
    """Consecutive trailing rounds whose best p_cal improved the running max
    by <= delta (round-aware; sampled siblings count into their round)."""
    rounds: dict[int, float] = {}
    for a in pool:
        if a.status == "success":
            rounds[a.round_index] = max(rounds.get(a.round_index, -1.0), a.p_cal)
    seq = [rounds[r] for r in sorted(rounds)]
    n, run_max = 0, seq[0] if seq else -1.0
    for v in seq[1:]:
        n = n + 1 if v <= run_max + delta else 0
        run_max = max(run_max, v)
    return n


def decide(pool: list[Arm], cfg: PolicyCfg, t: int) -> Action:
    """Pure decision over the current pool at round t (spec §4.1).

    Efficiency knobs (M6, dev-tuned, default off):
      tau0     — separate (lower) accept threshold at round 0: high-confidence
                 closed-book answers skip retrieval entirely (probe-router
                 evidence: q0-confident items answer closed-book at least as
                 well as with retrieval);
      patience — return the pool argmax once `patience` consecutive rounds
                 improved best p_cal by <= patience_delta (stagnation = the
                 remaining budget is unlikely to pay for itself)."""
    live = [a for a in pool if a.status == "success"]
    if not live:
        # nothing scoreable yet: keep going if budget remains, else abstain
        return Continue() if t < cfg.budget - 1 else Abstain()
    best = max(live, key=lambda a: a.p_cal)
    thr = cfg.tau0 if (cfg.tau0 is not None and t == 0) else cfg.tau_hi
    if best.p_cal >= thr:
        return Accept(best)
    if t >= cfg.budget - 1:
        if cfg.tau_lo is not None and best.p_cal < cfg.tau_lo:
            return Abstain()
        return ReturnArgmax(best)
    if cfg.patience is not None and t > 0 \
            and _stagnant_rounds(pool, cfg.patience_delta) >= cfg.patience:
        return ReturnArgmax(best)
    return Continue()


# --------------------------------------------------------------- runner ----
@dataclass
class ItemResult:
    question: str
    example_id: str
    answer: str
    action: str                 # accept | argmax | abstain
    stop_round: int
    pool: list[Arm] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question, "example_id": self.example_id,
            "answer": self.answer, "action": self.action, "stop_round": self.stop_round,
            "pool": [a.to_dict() for a in self.pool], "cost": self.cost.to_dict(),
            "diagnostics": self.diagnostics,
        }


class PGIR:
    """generate -> verify -> act loop over one retriever + ArmEvaluator."""

    def __init__(self, evaluator, retriever, cfg: PolicyCfg,
                 strategy: str = "s2", full_budget: bool = False):
        """full_budget=True disables early accept (val trace collection §8.1:
        thresholds only choose WHERE to stop; arms per round are unaffected,
        so a full-budget trace replays any (tau_hi, budget) offline)."""
        self.evaluator = evaluator
        self.retriever = retriever
        self.cfg = cfg
        self.strategy = strategy
        self.full_budget = full_budget

    def _expand_bon(self, base_arm: Arm, question: str, contexts, pids, t,
                    example_id: str) -> tuple[list[Arm], float]:
        """Adaptive best-of-N (SELECT): when the greedy arm lands in the
        uncertain band [bon_lo, tau_hi), sample bon_n-1 extra answers on the
        SAME evidence and score them; all enter the pool (traces then allow
        offline greedy-vs-bon replay). Returns (extra_arms, sampling_gen_s)."""
        cfg = self.cfg
        vllm = getattr(self.evaluator, "vllm", None)
        if cfg.bon_n <= 1 or vllm is None or base_arm.status != "success":
            return [], 0.0
        if not (cfg.bon_lo <= base_arm.p_cal < cfg.tau_hi):
            return [], 0.0
        t0 = time.monotonic()
        samples = vllm.generate_samples(question, contexts, n=cfg.bon_n - 1,
                                        temperature=cfg.bon_temperature)
        gen_s = time.monotonic() - t0
        seen = {" ".join(base_arm.answer.split()).casefold()}
        uniq = []
        for s in samples:
            key = " ".join(str(s).split()).casefold()
            if key and key not in seen:
                seen.add(key)
                uniq.append(str(s).strip())
        if not uniq:
            return [], gen_s
        jobs = [dict(question=question, contexts=contexts, answer=a,
                     answer_source="sampled", example_id=example_id or None,
                     evidence_pids=pids, round_index=t) for a in uniq]
        return self.evaluator.evaluate_many(jobs), gen_s

    def run_item(self, question: str, example_id: str = "") -> ItemResult:
        cfg = self.cfg
        total = Cost()
        state = EvidenceState()
        steer_log: list[str] = []
        bon_gen_s = 0.0

        def eval_arm(contexts, pids, t):
            arm = self.evaluator.evaluate(
                question, contexts=contexts, example_id=example_id or None,
                evidence_pids=pids, round_index=t)
            return arm

        def expand(base_arm, contexts, pids, t):
            nonlocal total, bon_gen_s
            extra, gen_s = self._expand_bon(base_arm, question, contexts, pids, t,
                                            example_id)
            bon_gen_s += gen_s
            for a in extra:
                total = total + a.cost
            return extra

        # round 0: closed-book arm (+ optional best-of-N siblings)
        pool = [eval_arm(None, [], 0)]
        total = total + pool[-1].cost
        pool += expand(pool[0], None, [], 0)
        act = decide(pool, cfg, 0)
        if isinstance(act, Accept) and not self.full_budget:
            return self._finalize(question, example_id, pool, act.arm, "accept", 0, total,
                                  state, steer_log=steer_log, bon_gen_s=bon_gen_s)

        # first evidence
        t0 = time.monotonic()
        docs = self.retriever.search(question, cfg.k)
        total.retr_calls += 1
        retr_log = [1]  # retriever calls made before each evidence round (replay §8.1)
        retr_times = [round(time.monotonic() - t0, 4)]
        state.register(docs)

        stop: tuple[Arm, str, int] | None = None
        for t in range(1, cfg.budget):
            if self.strategy == "s3" and t > 1:
                # S3: probe picks the best of m candidate evidence sets (§4.3)
                sets, n_retr = s3_candidates(question, pool, self.retriever, state, cfg)
                total.retr_calls += n_retr
                retr_log.append(n_retr)
                retr_times.append(0.0)
                jobs = [dict(question=question, contexts=render_contexts(s),
                             example_id=example_id or None,
                             evidence_pids=[d.get("id") for d in s], round_index=t)
                        for s in sets]
                arms = self.evaluator.evaluate_many(jobs)
                for a in arms:
                    total = total + a.cost
                live_arms = [a for a in arms if a.status == "success"] or arms
                best_arm = max(live_arms, key=lambda a: a.p_cal)
                docs = sets[arms.index(best_arm)]
                state.register(docs)
                pool.append(best_arm)
                pool += expand(best_arm, render_contexts(docs),
                               [d.get("id") for d in docs], t)
            else:
                contexts = render_contexts(docs)
                pool.append(eval_arm(contexts, [d.get("id") for d in docs], t))
                total = total + pool[-1].cost
                pool += expand(pool[-1], contexts, [d.get("id") for d in docs], t)
            act = decide(pool, cfg, t)
            if isinstance(act, Accept):
                stop = (act.arm, "accept", t)
                if not self.full_budget:
                    break
            elif isinstance(act, (ReturnArgmax, Abstain)):
                if stop is None:
                    arm = act.arm if isinstance(act, ReturnArgmax) else None
                    stop = (arm, "argmax" if arm is not None else "abstain", t)
                break
            if t < cfg.budget - 1 and self.strategy != "s3":
                t0 = time.monotonic()
                strategy, override_query = self.strategy, None
                if cfg.steer and self.strategy == "s2":
                    choice = decide_steer(pool, cfg)
                    if choice == "rewrite":
                        override_query = self._rewrite_query(question, pool, total)
                        if override_query is None:
                            choice = "s1"  # degenerate rewrite / no vllm: cheap fallback
                    if choice == "s1":
                        strategy = "s1"
                    steer_log.append(choice)
                docs, n_retr = next_evidence(
                    question, pool, docs, self.retriever, state, cfg, strategy,
                    override_query=override_query)
                total.retr_calls += n_retr
                retr_log.append(n_retr)
                retr_times.append(round(time.monotonic() - t0, 4))

        if stop is None or stop[0] is None and stop[1] != "abstain":
            live = [a for a in pool if a.status == "success"]
            best = max(live, key=lambda a: a.p_cal) if live else None
            stop = (best, "argmax" if best else "abstain", cfg.budget - 1)
        arm, action, t_stop = stop
        return self._finalize(question, example_id, pool, arm, action, t_stop, total, state,
                              retr_log, retr_times, steer_log=steer_log, bon_gen_s=bon_gen_s)

    def _rewrite_query(self, question: str, pool, total: Cost) -> str | None:
        """STEER rewrite: one free-form gen call -> alternative retrieval query.
        Wall time is charged to the surrounding retr_times entry."""
        vllm = getattr(self.evaluator, "vllm", None)
        if vllm is None:
            return None
        wrong = [a.answer for a in pool if a.status == "success" and a.answer]
        try:
            raw = vllm.generate_free(build_rewrite_prompt(question, wrong),
                                     max_tokens=REWRITE_MAX_TOKENS)
        except Exception:
            return None  # rewrite is best-effort; never fail the item on it
        total.gen_calls += 1
        return clean_rewrite(raw, question) or None

    def _finalize(self, question, example_id, pool, arm, action, t, total, state,
                  retr_log=None, retr_times=None, steer_log=None,
                  bon_gen_s: float = 0.0) -> ItemResult:
        diag = state.diagnostics(pool)
        diag["retr_log"] = list(retr_log or [])
        diag["retr_times_s"] = list(retr_times or [])
        diag["steer_log"] = list(steer_log or [])
        # concurrency-independent per-item latency: this item's own calls, summed
        # (bon_gen_s = wall time of the batched sampling requests, not per-arm)
        diag["serial_latency_s"] = round(
            sum(a.serial_s for a in pool) + sum(retr_times or []) + bon_gen_s, 4)
        return ItemResult(
            question=question, example_id=example_id,
            answer=(arm.answer if arm is not None else ""),
            action=action, stop_round=t, pool=pool, cost=total,
            diagnostics=diag,
        )
