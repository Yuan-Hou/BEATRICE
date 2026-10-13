"""Recompute the PGIR budget-frontier Pareto points (B=2/3/4 @ tau_hi) by
offline replay of the six test traces, macro-averaged; writes
runs/tables/pgir_pareto_points.json (consumed by scripts/plot_pareto.py, which
keeps its own hardcoded copy — update it from this output).

B<4 replay from a tau=0.6/B=4 run is exact: early stopping only ever consumes
a prefix of the recorded rounds.

Run: cd <package_root> && python -m pgir.scripts.make_pareto_points \
         --trace-root pgir/runs/flashrag/m7_fused
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgir.pipeline import PolicyCfg
from pgir.tuning.trace import em, f1, load_golds, load_trace, simulate

PGIR_DIR = Path(__file__).resolve().parent.parent
DATASETS = ["hotpotqa", "twowiki", "musique", "popqa", "nq_ood", "triviaqa_ood"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", required=True, nargs="+",
                    help="dir(s) holding {ds}_*/pgir_trace_{ds}_test.jsonl; when a "
                         "dataset appears under several roots the newest trace wins")
    ap.add_argument("--tau-hi", type=float, default=0.6)
    # budget=1 -> closed-book-only anchor (zero retrieval rounds; the user's
    # "B=0 retrievals"); budget=2 -> at most one retrieval ("B=1"). Recorded
    # for the record even if only 2/3/4 make the figure.
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", default=str(PGIR_DIR / "runs" / "tables" / "pgir_pareto_points.json"))
    args = ap.parse_args()

    golds = load_golds(PGIR_DIR / "datasets")
    roots = [Path(r) for r in args.trace_root]
    per_ds = {}
    for ds in DATASETS:
        fs = [p for root in roots
              for p in root.glob(f"{ds}_*/pgir_trace_{ds}_test.jsonl")]
        fs.sort(key=lambda p: p.stat().st_mtime)
        if not fs:
            raise SystemExit(f"missing test trace for {ds} under {roots}")
        per_ds[ds] = load_trace(fs[-1])
        print(f"{ds}: {len(per_ds[ds])} items ({fs[-1]})")

    points = {}
    for b in args.budgets:
        cfg = PolicyCfg(tau_hi=args.tau_hi, budget=b)
        ems, f1s, gens = [], [], []
        for ds, traces in per_ds.items():
            sims = [simulate(it, cfg) for it in traces]  # stored p_cal
            n = len(sims)
            ems.append(sum(em(s["answer"], golds[s["example_id"]]) for s in sims) / n)
            f1s.append(sum(f1(s["answer"], golds[s["example_id"]]) for s in sims) / n)
            gens.append(sum(s["gen_calls"] for s in sims) / n)
        k = len(per_ds)
        tag = f"B{b}" + (" (main)" if b == max(args.budgets) else "")
        points[tag] = {"em": sum(ems) / k, "f1": sum(f1s) / k, "gen": sum(gens) / k}
        print(f"{tag}: EM={points[tag]['em']:.4f} F1={points[tag]['f1']:.4f} "
              f"gen={points[tag]['gen']:.3f}")

    Path(args.out).write_text(json.dumps(points, indent=1, ensure_ascii=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
