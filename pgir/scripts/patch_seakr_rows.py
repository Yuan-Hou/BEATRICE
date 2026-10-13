"""Re-apply the FAITHFUL SeaKR rows to runs/tables/main_full_metrics.md.

rebuild_tables regenerates the whole table from runs/flashrag/* dirs, but the
faithful SeaKR reproduction (official THU-KEG reasoner + Qwen3.5 engine,
ES-BM25, theta=0.0, 2026-07-10) lives in ~/SeaKR/runs/*.jsonl — so a rebuild
silently reverts seakr to the old method-level repro. Run this AFTER every
rebuild:  cd <package_root> && python -m pgir.scripts.patch_seakr_rows
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from pgir.analysis.rebuild_tables import load_scope, prf  # same normalizer as the table

SEAKR = Path("SeaKR/runs")
TABLES = Path("pgir/runs/tables")
DATASETS = ["hotpotqa", "twowiki", "musique", "popqa", "nq_ood", "triviaqa_ood"]


def dataset_row(ds: str, keep: set | None = None) -> tuple[str, dict]:
    f = SEAKR / f"seakr_qwen_{ds}_test_th0.0.jsonl"
    recs, bad = [], 0
    with open(f, encoding="utf-8") as fh:  # NOT splitlines(): answers may contain
        for l in fh:                       #  -style unicode line separators
            if not l.strip():
                continue
            try:
                recs.append(json.loads(l))
            except json.JSONDecodeError:
                bad += 1  # torn tail line from a killed writer; resumable
    if bad:
        print(f"[warn] {f.name}: skipped {bad} torn line(s)")
    if keep is not None:
        recs = [r for r in recs if r.get("id") in keep]
    stats = {"n": len(recs), "em": 0.0, "f1": 0.0, "precision": 0.0,
             "recall": 0.0, "gen": 0.0}
    for r in recs:
        golds = r["golden"]
        if isinstance(golds, str):
            golds = ast.literal_eval(golds)
        m = prf(r["answer"], [str(g) for g in golds])
        for k in ("em", "f1", "precision", "recall"):
            stats[k] += m[k]
        stats["gen"] += r.get("llm_calls", 0)
    n = stats["n"]
    for k in ("em", "f1", "precision", "recall", "gen"):
        stats[k] /= max(n, 1)
    line = (f"| seakr | {ds} | {n} | {stats['em']:.3f} | {stats['f1']:.3f} | "
            f"{stats['precision']:.3f} | {stats['recall']:.3f} | {round(stats['gen'], 2)} |")
    return line, stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default=None, help="scope-ids json: re-slice to the scoped subset")
    ap.add_argument("--suffix", default=None,
                    help="table filename suffix (default '_scoped' when --scope given)")
    args = ap.parse_args()
    scope = load_scope(args.scope) if args.scope else None
    suffix = args.suffix if args.suffix is not None else ("_scoped" if scope else "")
    md_path = TABLES / f"main_full_metrics{suffix}.md"

    rows, agg = [], []
    for ds in DATASETS:
        line, s = dataset_row(ds, keep=scope.get(ds) if scope else None)
        rows.append(line)
        agg.append(s)
    k = len(agg)
    rows.append(f"| seakr | **MACRO** | — | {sum(s['em'] for s in agg)/k:.3f} | "
                f"{sum(s['f1'] for s in agg)/k:.3f} | {sum(s['precision'] for s in agg)/k:.3f} | "
                f"{sum(s['recall'] for s in agg)/k:.3f} | {sum(s['gen'] for s in agg)/k:.2f} |")
    text = md_path.read_text()
    kept = [l for l in text.splitlines() if not re.match(r"\|\s*seakr\s*\|", l)]
    # seakr sits between dragin and pgir in CORE_METHODS order
    out, inserted = [], False
    for l in kept:
        if not inserted and re.match(r"\|\s*pgir\s*\|", l):
            out.extend(rows)
            inserted = True
        out.append(l)
    if not inserted:
        out.extend(rows)
    md_path.write_text("\n".join(out) + "\n")
    print("\n".join(rows))
    print(f"patched {md_path}")


if __name__ == "__main__":
    main()
