"""M2/M3 smoke table: every system x every dataset on a small subset.

Runs run_baselines for each (method, dataset) combo sequentially (they share
the vLLM server and the in-RAM index), collects FlashRAG metric outputs +
cost summaries into one markdown table under runs/smoke_table.md.

Run: cd <package_root> && python -m pgir.run_smoke_table [-n 200] [--split dev]
     [--methods norag vanilla ...] [--datasets hotpotqa ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

PGIR_DIR = Path(__file__).resolve().parent
RUNS = PGIR_DIR / "runs" / "flashrag"

DEFAULT_METHODS = ["norag", "vanilla", "ircot", "iterretgen", "flare",
                   "skr", "selfask", "sure", "pgir"]
DEFAULT_DATASETS = ["hotpotqa", "twowiki", "musique", "popqa"]  # + nq/triviaqa via --datasets


def latest_metrics(save_dir: Path) -> dict:
    """FlashRAG writes <save_dir>/<timestamp>_<note>/metric_score.txt + intermediate."""
    cands = sorted(save_dir.glob("**/metric_score.txt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return {}
    metrics = {}
    for line in cands[-1].read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                metrics[k.strip()] = float(v)
            except ValueError:
                pass
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200, help="<=0 means the full split")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--stop-on-error", action="store_true")
    # pgir policy (locked on dev by tuning/run_tuning.py before any test run)
    ap.add_argument("--tau-hi", type=float, default=0.8)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--calibrator", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.n <= 0:
        args.n = None
    if args.split == "test":
        print("[note] smoke on TEST split — only do this for the final run!")

    rows = []
    for method in args.methods:
        for ds in args.datasets:
            if ds == "popqa" and args.split == "dev":
                continue  # popqa has no dev (OOD test only)
            note = f"smoke_{method}_{ds}_{args.split}_n{args.n or 'full'}"
            save_dir = RUNS / note
            print(f"\n=== {method} / {ds} ===")
            t0 = time.time()
            try:
                from pgir.run_baselines import run_one
                kw = {}
                if method == "pgir":
                    kw = dict(tau_hi=args.tau_hi, budget=args.budget,
                              calibrator=args.calibrator, workers=args.workers)
                run_one(method, ds, args.split, sample_num=args.n, save_note=note, **kw)
                ok = True
            except Exception:
                ok = False
                traceback.print_exc()
                if args.stop_on_error:
                    sys.exit(1)
            metrics = latest_metrics(save_dir) if ok else {}
            cost = {}
            # FlashRAG rewrites save_dir to a timestamped subdir — search for it
            cands = sorted(save_dir.glob("**/cost_summary.json"),
                           key=lambda p: p.stat().st_mtime)
            if cands:
                cost = json.loads(cands[-1].read_text())
            rows.append({
                "method": method, "dataset": ds, "ok": ok,
                "em": metrics.get("em"), "f1": metrics.get("f1"),
                "gen/item": cost.get("gen_calls_per_item"),
                "retr/item": cost.get("retr_calls_per_item"),
                "wall_s": round(time.time() - t0, 1),
            })
            tag = f"{args.split}_n{args.n or 'full'}"
            (RUNS / f"table_{tag}.json").write_text(json.dumps(rows, indent=2))

    # markdown table
    md = [f"# Smoke table (split={args.split}, n={args.n})", "",
          "| method | dataset | ok | EM | F1 | gen/item | retr/item | wall(s) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {method} | {dataset} | {ok} | {em} | {f1} | {g} | {r_} | {w} |".format(
            method=r["method"], dataset=r["dataset"], ok=r["ok"],
            em=("%.4f" % r["em"]) if r["em"] is not None else "-",
            f1=("%.4f" % r["f1"]) if r["f1"] is not None else "-",
            g=("%.2f" % r["gen/item"]) if r["gen/item"] is not None else "-",
            r_=("%.2f" % r["retr/item"]) if r["retr/item"] is not None else "-",
            w=r["wall_s"]))
    (RUNS / f"table_{args.split}_n{args.n or 'full'}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
