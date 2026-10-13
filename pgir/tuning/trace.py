"""Offline replay of full-budget PGIR traces + threshold grid sweep (spec §8).

A full-budget trace fixes WHAT each round generates; (tau_hi, budget, tau_lo)
only choose WHERE to stop — so any policy is replayable offline for free.
`simulate` mirrors `pipeline.decide` exactly (test_simulate_consistency).

EM/F1 reuse FlashRAG's own normalize/metric logic.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from flashrag.evaluator.utils import normalize_answer

from pgir.pipeline import PolicyCfg


# ---- FlashRAG-equivalent EM / token-F1 on one prediction --------------------
def em(prediction: str, golden_answers: list[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(g) for g in golden_answers))


def f1(prediction: str, golden_answers: list[str]) -> float:
    best = 0.0
    for g in golden_answers:
        np_, ng = normalize_answer(prediction), normalize_answer(g)
        if np_ in ["yes", "no", "noanswer"] and np_ != ng:
            continue
        if ng in ["yes", "no", "noanswer"] and np_ != ng:
            continue
        pt, gt = np_.split(), ng.split()
        same = sum((Counter(pt) & Counter(gt)).values())
        if same == 0:
            continue
        prec, rec = same / len(pt), same / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def load_trace(path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def _p(arm: dict, calibrator=None) -> float:
    if calibrator is not None:
        return float(calibrator.calibrate(arm["p_raw"], arm["mode"]))
    return float(arm["p_cal"])


def simulate(item: dict, cfg: PolicyCfg, calibrator=None) -> dict:
    """Replay one full-budget trace item under a policy. Mirrors decide().

    Round-index aware: with best-of-N traces a round holds several arms
    (greedy + sampled siblings); decide() sees them all at that round's
    boundary. Single-arm-per-round traces replay exactly as before."""
    pool = [a for a in item["pool"] if a["round_index"] < cfg.budget]
    retr_log = list(item.get("diagnostics", {}).get("retr_log", []))
    rounds = sorted({a["round_index"] for a in pool})
    live, seen, stop = [], [], None
    round_best: list[float] = []  # per-round best p (patience mirror of decide)
    for t in rounds:
        round_arms = [a for a in pool if a["round_index"] == t]
        seen.extend(round_arms)
        live.extend(a for a in round_arms if a["status"] == "success")
        ok = [a for a in round_arms if a["status"] == "success"]
        round_best.append(max((_p(a, calibrator) for a in ok), default=-1.0))
        if live:
            best = max(live, key=lambda a: _p(a, calibrator))
            thr = cfg.tau0 if (cfg.tau0 is not None and t == 0) else cfg.tau_hi
            if _p(best, calibrator) >= thr:
                stop = (best, "accept", t)
                break
            if t >= cfg.budget - 1:
                if cfg.tau_lo is not None and _p(best, calibrator) < cfg.tau_lo:
                    stop = (None, "abstain", t)
                else:
                    stop = (best, "argmax", t)
                break
            if cfg.patience is not None and t > 0:
                n, run_max = 0, round_best[0]
                for v in round_best[1:]:
                    n = n + 1 if v <= run_max + cfg.patience_delta else 0
                    run_max = max(run_max, v)
                if n >= cfg.patience:
                    stop = (best, "argmax", t)
                    break
        elif t >= cfg.budget - 1:
            stop = (None, "abstain", t)
            break
    if stop is None:  # trace shorter than budget: pool exhausted
        t = rounds[-1] if rounds else 0
        best = max(live, key=lambda a: _p(a, calibrator)) if live else None
        stop = (best, "argmax" if best else "abstain", t)
    arm, action, t = stop
    n_arms = len(seen)  # every arm consumed 1 gen + 1 score
    retr_times = item.get("diagnostics", {}).get("retr_times_s", [])
    serial_s = (sum(float(a.get("timing", {}).get("gen_s", 0.0))
                    + float(a.get("timing", {}).get("score_s", 0.0))
                    for a in seen)
                + sum(retr_times[:t]))
    return {
        "example_id": item["example_id"],
        "answer": arm["answer"] if arm else "",
        "action": action, "stop_round": t,
        "gen_calls": n_arms, "score_calls": n_arms,
        # lazy costing: round 0 needs no retrieval; round t consumed retr_log[:t]
        "retr_calls": sum(retr_log[:t]),
        "retr_calls_prefetch": sum(retr_log[:max(t, 1)]),
        "serial_latency_s": round(serial_s, 4),
    }


def sweep(traces: list[dict], golds: dict[str, list[str]],
          tau_his=(0.6, 0.7, 0.8, 0.9, 0.95), budgets=(2, 3, 4),
          calibrator=None, tau_lo=None) -> list[dict]:
    rows = []
    for budget in budgets:
        for tau_hi in tau_his:
            cfg = PolicyCfg(tau_hi=tau_hi, budget=budget, tau_lo=tau_lo)
            sims = [simulate(it, cfg, calibrator) for it in traces]
            n = len(sims)
            rows.append({
                "tau_hi": tau_hi, "budget": budget,
                "em": sum(em(s["answer"], golds[s["example_id"]]) for s in sims) / n,
                "f1": sum(f1(s["answer"], golds[s["example_id"]]) for s in sims) / n,
                "gen_calls": sum(s["gen_calls"] for s in sims) / n,
                "score_calls": sum(s["score_calls"] for s in sims) / n,
                "retr_calls": sum(s["retr_calls"] for s in sims) / n,
                "accept_rate": sum(s["action"] == "accept" for s in sims) / n,
            })
    return rows


def select_config(rows: list[dict], max_gen_calls: float) -> dict:
    """Spec §6-5: best val EM subject to mean #gen_calls <= constraint."""
    ok = [r for r in rows if r["gen_calls"] <= max_gen_calls]
    pool = ok or rows  # if nothing satisfies, fall back to cheapest-then-best
    if not ok:
        min_gen = min(r["gen_calls"] for r in rows)
        pool = [r for r in rows if r["gen_calls"] == min_gen]
    return max(pool, key=lambda r: (r["em"], -r["gen_calls"]))


def load_golds(dataset_dir: str | Path) -> dict[str, list[str]]:
    """example_id -> golden_answers from the pgir/datasets FlashRAG files."""
    golds = {}
    for f in Path(dataset_dir).rglob("*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            golds[r["id"]] = r["golden_answers"]
    return golds
