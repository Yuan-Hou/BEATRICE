"""PGIR signature analyses over traces (spec §8.4).

- online_regression_rate: final answer wrong while the round-0 (closed-book)
  answer was right — the failure PGIR's pool-argmax is designed to prevent;
- oracle_headroom_captured: (PGIR - NoRAG) / (pool-oracle - NoRAG);
- lock-in diagnostics aggregation (evidence overlap, forced S1, plateaus).

All functions take a trace (list of item dicts) + golds and a `final` map
(example_id -> final answer) from either a live run or tuning.trace.simulate.
"""
from __future__ import annotations

from pgir.tuning.trace import em, f1


def _round0(item: dict) -> dict | None:
    for a in item["pool"]:
        if a["round_index"] == 0 and a["status"] == "success":
            return a
    return None


def online_regression_rate(traces: list[dict], golds: dict, final: dict) -> dict:
    n, regressed, round0_right = 0, 0, 0
    for it in traces:
        r0 = _round0(it)
        if r0 is None:
            continue
        n += 1
        g = golds[it["example_id"]]
        r0_em = em(r0["answer"], g)
        fin_em = em(final.get(it["example_id"], ""), g)
        round0_right += r0_em
        if r0_em == 1.0 and fin_em == 0.0:
            regressed += 1
    return {
        "n": n,
        "online_regression_rate": regressed / n if n else 0.0,
        "round0_em": round0_right / n if n else 0.0,
    }


def oracle_headroom_captured(traces: list[dict], golds: dict, final: dict,
                             metric=None) -> dict:
    metric = metric or em
    sys_score = norag = oracle = 0.0
    n = 0
    for it in traces:
        g = golds[it["example_id"]]
        r0 = _round0(it)
        pool = [a for a in it["pool"] if a["status"] == "success"]
        if not pool:
            continue
        n += 1
        sys_score += metric(final.get(it["example_id"], ""), g)
        norag += metric(r0["answer"], g) if r0 else 0.0
        oracle += max(metric(a["answer"], g) for a in pool)  # hindsight-best arm
    if n == 0:
        return {"n": 0}
    sys_score, norag, oracle = sys_score / n, norag / n, oracle / n
    denom = oracle - norag
    return {
        "n": n, "system": sys_score, "norag_round0": norag, "pool_oracle": oracle,
        "headroom_captured": (sys_score - norag) / denom if denom > 1e-9 else float("nan"),
    }


def lockin_diagnostics(traces: list[dict]) -> dict:
    n = len(traces)
    if n == 0:
        return {"n": 0}
    overlap = [it["diagnostics"].get("evidence_overlap_rate", 0.0) for it in traces]
    plateau = [it["diagnostics"].get("score_plateau_rate", 0.0) for it in traces]
    forced = [it["diagnostics"].get("forced_s1", 0) for it in traces]
    return {
        "n": n,
        "mean_evidence_overlap_rate": sum(overlap) / n,
        "mean_score_plateau_rate": sum(plateau) / n,
        "forced_s1_per_item": sum(forced) / n,
    }


def per_round_gain_curve(traces: list[dict], golds: dict, metric=None) -> list[dict]:
    """Best-so-far pool metric after each round — S2 recall-growth evidence."""
    metric = metric or f1
    max_rounds = max((len(it["pool"]) for it in traces), default=0)
    rows = []
    for t in range(max_rounds):
        vals, n = 0.0, 0
        for it in traces:
            pool = [a for a in it["pool"] if a["status"] == "success"
                    and a["round_index"] <= t]
            if not pool:
                continue
            n += 1
            vals += max(metric(a["answer"], golds[it["example_id"]]) for a in pool)
        rows.append({"round": t, "pool_best": vals / n if n else float("nan"), "n": n})
    return rows
