"""Unified AUROC/AUPRC evaluator for any prediction JSONL(s).

Works on baseline outputs *and* on the main probe's prediction files (frozen fusion,
resnet, ...), since they all share the ``{label, helpful_probability, dataset}`` schema.
Use it to put every method in one table on identical samples/metrics.

Usage:
    python scripts/evaluate_predictions.py \
        --preds outputs/predictions/frozen_fusion_question_validation_preds.jsonl \
                outputs/predictions/baselines/*.jsonl \
        --threshold 0.5 [--by-dataset] [--output metrics.json]
"""

from __future__ import annotations

import argparse
import glob
import json

from ragprobe.baselines.evaluate import evaluate_prediction_files, format_report_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", nargs="+", required=True, help="Prediction JSONL paths (globs allowed).")
    parser.add_argument("--score-key", default=None, help="Score field; default auto-detects helpful_probability/score.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--by-dataset", action="store_true", help="Also print per-dataset AUROC.")
    parser.add_argument("--output", default=None, help="Optional path to write the full metrics JSON.")
    args = parser.parse_args()

    paths = []
    for pat in args.preds:
        expanded = sorted(glob.glob(pat))
        paths.extend(expanded if expanded else [pat])

    report = evaluate_prediction_files(paths, score_key=args.score_key, threshold=args.threshold)
    print(format_report_table(report))

    if args.by_dataset:
        for name, res in report.items():
            print(f"\n# {name}")
            for ds, m in res["by_dataset"].items():
                auroc = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "n/a"
                auprc = f"{m['auprc']:.4f}" if m.get("auprc") is not None else "n/a"
                print(f"  {ds:24s} n={m['n']:5d} pos={m['positive_rate']*100:4.1f}% AUROC={auroc} AUPRC={auprc}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
