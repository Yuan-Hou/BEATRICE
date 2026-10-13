"""Compute per-example Semantic Entropy (NLI-clustered) from the sampling pass and
cache it to disk, so the faithful SEP probe (train SE labels) and the SE baseline
(eval SE) read identical, reproducible values.

Writes data/se_samples/<setting>/<group>.se.jsonl with one row per example:
    {example_id, dataset, split, label, discrete_se, proper_se, num_clusters, k}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragprobe.baselines.semantic_entropy import compute_se_for_samples
from ragprobe.utils.io import read_jsonl
from ragprobe.utils.logging import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--samples-dir", default=None)
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="Only (re)compute SE for these group files; default all.")
    args = ap.parse_args()

    sdir = Path(args.samples_dir or f"data/se_samples/{args.setting}")
    for path in sorted(sdir.glob("*.jsonl")):
        if path.name.endswith(".se.jsonl"):
            continue
        group = path.stem
        if args.groups and group not in args.groups:
            continue
        rows = list(read_jsonl(path))
        log(f"[{args.setting}/{group}] computing SE for {len(rows)} examples (nli={not args.no_nli})...")
        recs = compute_se_for_samples(rows, use_nli=not args.no_nli, nli_model=args.nli_model)
        out = path.with_suffix(".se.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({
                    "example_id": r.example_id, "dataset": r.dataset, "split": r.split,
                    "label": int(r.label), "discrete_se": r.discrete_se,
                    "proper_se": r.proper_se, "num_clusters": r.num_clusters, "k": r.k,
                }, ensure_ascii=False) + "\n")
        log(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
