"""Greedy-vs-best-of-N offline replay + sampling sanity checks (M6).

Traces collected with --bon-n K contain, per round, the greedy arm plus up to
K-1 sampled siblings (answer_source="sampled"). Stripping the sampled arms
reproduces the legacy greedy pool exactly, so ONE trace run yields both
policies offline:

  em_bon - em_greedy      = what best-of-N buys at each (tau_hi, budget)
  pool-ceiling lift        = items where only a SAMPLED arm is EM-correct
                             (the metric that motivated M6: dev oracle said
                             the pool never contains a right answer on
                             .55-.89 of misses)
  sampled-arm AUROC        = does the probe still rank answers it did not
                             greedy-decode? (probe was trained on greedy
                             answers — distribution check before trusting)

Run AFTER tuning/run_tuning.py (reuses its calibrator):
  cd <package_root> && python -m pgir.tuning.bon_replay \
      --traces pgir/runs/flashrag/*/pgir_trace_*dev*.jsonl \
      --calibrator pgir/runs/tuning/calibrator.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pgir.calibration import PerModeCalibrator
from pgir.tuning.run_tuning import auroc
from pgir.tuning.trace import em, load_golds, load_trace, sweep

PGIR_DIR = Path(__file__).resolve().parent.parent


def strip_sampled(item: dict) -> dict:
    out = copy.copy(item)
    out["pool"] = [a for a in item["pool"] if a.get("answer_source") != "sampled"]
    return out


def band_filter(item: dict, lo: float, hi: float, calibrator=None) -> dict:
    """Keep sampled arms only in rounds whose GREEDY arm scores inside
    [lo, hi) — replays a narrower bon_lo/tau_hi trigger band offline.
    (Collection-time band was [0.05, 0.8); narrower bands are a subset.)"""
    from pgir.tuning.trace import _p
    greedy_p = {a["round_index"]: _p(a, calibrator) for a in item["pool"]
                if a.get("answer_source") != "sampled" and a["status"] == "success"}
    out = copy.copy(item)
    out["pool"] = [a for a in item["pool"]
                   if a.get("answer_source") != "sampled"
                   or lo <= greedy_p.get(a["round_index"], -1.0) < hi]
    return out


def sanity(traces, golds) -> dict:
    g_p, g_y, s_p, s_y = [], [], [], []
    n_rounds = n_fired = n_sampled_arms = 0
    ceiling_base = ceiling_lift = 0
    for it in traces:
        gold = golds[it["example_id"]]
        rounds: dict[int, list] = {}
        for a in it["pool"]:
            rounds.setdefault(a["round_index"], []).append(a)
        greedy_ok = sampled_ok = False
        for arms in rounds.values():
            n_rounds += 1
            sampled = [a for a in arms if a.get("answer_source") == "sampled"]
            if sampled:
                n_fired += 1
                n_sampled_arms += len(sampled)
            for a in arms:
                if a["status"] != "success":
                    continue
                y = int(em(a["answer"], gold))
                if a.get("answer_source") == "sampled":
                    s_p.append(float(a["p_raw"])); s_y.append(y)
                    sampled_ok = sampled_ok or bool(y)
                else:
                    g_p.append(float(a["p_raw"])); g_y.append(y)
                    greedy_ok = greedy_ok or bool(y)
        if not greedy_ok:
            ceiling_base += 1
            if sampled_ok:
                ceiling_lift += 1
    return {
        "n_items": len(traces), "n_rounds": n_rounds,
        "bon_fire_rate": n_fired / max(n_rounds, 1),
        "sampled_arms_per_fired_round": n_sampled_arms / max(n_fired, 1),
        "n_greedy_arms": len(g_p), "n_sampled_arms": len(s_p),
        "greedy_em_rate": sum(g_y) / max(len(g_y), 1),
        "sampled_em_rate": sum(s_y) / max(len(s_y), 1),
        "auroc_greedy_arms": auroc(g_p, g_y) if g_p else float("nan"),
        "auroc_sampled_arms": auroc(s_p, s_y) if s_p else float("nan"),
        # items with NO correct greedy arm, and how many of those a sampled arm rescues
        "items_no_correct_greedy": ceiling_base,
        "items_rescued_by_sampling": ceiling_lift,
        "pool_ceiling_lift": ceiling_lift / max(ceiling_base, 1),
    }


def compare(traces, golds, calibrator, tau_his, budgets) -> list[dict]:
    greedy_traces = [strip_sampled(it) for it in traces]
    bon_rows = sweep(traces, golds, tau_his, budgets, calibrator)
    grd_rows = sweep(greedy_traces, golds, tau_his, budgets, calibrator)
    rows = []
    for b, g in zip(bon_rows, grd_rows):
        assert (b["tau_hi"], b["budget"]) == (g["tau_hi"], g["budget"])
        rows.append({
            "tau_hi": b["tau_hi"], "budget": b["budget"],
            "em_greedy": g["em"], "em_bon": b["em"],
            "delta_em": round(b["em"] - g["em"], 4),
            "gen_greedy": g["gen_calls"], "gen_bon": b["gen_calls"],
            "f1_greedy": g["f1"], "f1_bon": b["f1"],
            "accept_rate_bon": b["accept_rate"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--calibrator", default=None,
                    help="runs/tuning/calibrator.json from run_tuning (optional: raw p_cal otherwise)")
    ap.add_argument("--tau-his", nargs="+", type=float,
                    default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--budgets", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--bands", nargs="+", default=[],
                    help="narrower bon trigger bands to replay, e.g. 0.2:0.7 0.3:0.6")
    ap.add_argument("--band-tau", type=float, default=0.7,
                    help="tau_hi used for the band scan")
    ap.add_argument("--band-budget", type=int, default=4)
    ap.add_argument("--out", default=str(PGIR_DIR / "runs" / "tuning" / "bon_replay"))
    args = ap.parse_args()

    golds = load_golds(PGIR_DIR / "datasets")
    cal = PerModeCalibrator.load(args.calibrator) if args.calibrator else None

    per_file, all_traces = {}, []
    for t in args.traces:
        part = load_trace(t)
        all_traces.extend(part)
        per_file[Path(t).name] = part

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"sanity_all": sanity(all_traces, golds),
              "compare_all": compare(all_traces, golds, cal,
                                     tuple(args.tau_his), tuple(args.budgets)),
              "sanity_per_file": {k: sanity(v, golds) for k, v in per_file.items()}}

    if args.bands:
        band_rows = []
        for spec in args.bands:
            lo, hi = (float(x) for x in spec.split(":"))
            banded = [band_filter(it, lo, hi, cal) for it in all_traces]
            r = compare(banded, golds, cal, (args.band_tau,), (args.band_budget,))[0]
            band_rows.append({"band": spec, **r})
        result["band_scan"] = band_rows
        print(f"band scan @ tau={args.band_tau} B={args.band_budget} "
              f"(greedy em={band_rows[0]['em_greedy']:.3f} gen={band_rows[0]['gen_greedy']:.2f}):")
        for r in band_rows:
            print(f"  band {r['band']:>9}: em_bon={r['em_bon']:.3f} "
                  f"dEM={r['delta_em']:+.3f} gen_bon={r['gen_bon']:.2f}")

    (out / "bon_replay.json").write_text(json.dumps(result, indent=2))

    s = result["sanity_all"]
    print(f"sampling: fire_rate={s['bon_fire_rate']:.2f} "
          f"arms/fired={s['sampled_arms_per_fired_round']:.2f} "
          f"sampled_em={s['sampled_em_rate']:.3f} vs greedy_em={s['greedy_em_rate']:.3f}")
    print(f"probe on sampled answers: AUROC={s['auroc_sampled_arms']:.3f} "
          f"(greedy arms {s['auroc_greedy_arms']:.3f})")
    print(f"pool-ceiling lift: sampling rescues {s['items_rescued_by_sampling']}"
          f"/{s['items_no_correct_greedy']} previously-hopeless items "
          f"({s['pool_ceiling_lift']:.1%})")
    print(f"{'tau':>5} {'B':>2} {'em_grd':>7} {'em_bon':>7} {'dEM':>7} "
          f"{'gen_grd':>8} {'gen_bon':>8}")
    for r in result["compare_all"]:
        print(f"{r['tau_hi']:>5} {r['budget']:>2} {r['em_greedy']:>7.3f} "
              f"{r['em_bon']:>7.3f} {r['delta_em']:>+7.3f} "
              f"{r['gen_greedy']:>8.2f} {r['gen_bon']:>8.2f}")
    print(f"-> {out}/bon_replay.json")


if __name__ == "__main__":
    main()
