"""M3 dev trace collection: PGIR full-budget over all dev datasets, one process
(index/corpus loaded once). Traces feed run_tuning (calibration + sweep).

Run: cd <package_root> && python -m pgir.tuning.collect_traces [--workers 12] [-n N]
"""
from __future__ import annotations

import argparse
import json
import time

DATASETS = ("hotpotqa", "twowiki", "musique")  # popqa has no dev (OOD test only)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("-n", type=int, default=None, help="items per dataset (default all 500)")
    ap.add_argument("--budget", type=int, default=4)
    args = ap.parse_args()

    from pgir.run_baselines import run_one

    for ds in DATASETS:
        note = f"trace_{ds}_dev_full" + (f"_n{args.n}" if args.n else "")
        t0 = time.time()
        summary = run_one("pgir", ds, "dev", sample_num=args.n, budget=args.budget,
                          full_budget=True, workers=args.workers, save_note=note)
        print(f"[{ds}] done in {time.time()-t0:.0f}s: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
