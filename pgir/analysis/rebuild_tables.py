"""Rebuild all result tables with full metrics (EM / F1 / Precision / Recall)
from the on-disk predictions — no reruns needed.

Scans runs/flashrag/smoke_*_test_nfull (main table), smoke_*_test_n500
(generalization) and s3_ablation_* dirs, recomputes token-level metrics from
each run's intermediate_data.json with the FlashRAG-identical normalizer,
and writes runs/tables/{main,generalization,s3}_full_metrics.{json,md}.

Run: cd <package_root> && python -m pgir.analysis.rebuild_tables
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from flashrag.evaluator.utils import normalize_answer

RUNS = Path("pgir/runs/flashrag")
OUT = Path("pgir/runs/tables")

# Main-table scope: the primary comparison keeps only the confidence-control
# family (methods on the same control axis as PGIR); cross-paradigm systems
# (interleaved CoT retrieval, generation-augmented iteration, question
# decomposition, summary-candidate voting) go to the paradigm_ref reference
# tables. All numbers are kept; per-dataset overtake points are disclosed.
# FLARE dropped entirely (user 2026-07-08): EM~0 at every threshold, no
# reference value; run data kept under runs/flashrag/smoke_flare_* only.
CORE_METHODS = ["norag", "vanilla", "skr", "adaptive",
                "probingrag", "dragin", "seakr", "pgir"]
PARADIGM_REF_METHODS = ["ircot", "iterretgen", "selfask", "sure"]
MAIN_METHODS = CORE_METHODS + PARADIGM_REF_METHODS  # legacy alias (all methods)
# main table = 4 canonical test sets + BOTH user-sampled OOD sets (user
# 2026-07-08: one complete table, all datasets, all metrics)
MAIN_DATASETS = ["hotpotqa", "twowiki", "musique", "popqa",
                 "nq_ood", "triviaqa_ood"]
GEN_DATASETS = ["nq", "triviaqa"]  # legacy native-split n500 columns


def prf(prediction: str, golds: list[str]) -> dict:
    """Token-level P/R/F1 (max over golds) + EM, FlashRAG-identical."""
    np_ = normalize_answer(prediction)
    em = float(any(np_ == normalize_answer(g) for g in golds))
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    for g in golds:
        ng = normalize_answer(g)
        if np_ in ["yes", "no", "noanswer"] and np_ != ng:
            continue
        if ng in ["yes", "no", "noanswer"] and np_ != ng:
            continue
        pt, gt = np_.split(), ng.split()
        same = sum((Counter(pt) & Counter(gt)).values())
        if same == 0:
            continue
        p, r = same / len(pt), same / len(gt)
        f1 = 2 * p * r / (p + r)
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": p, "recall": r}
    return {"em": em, **best}


def load_scope(scope_path) -> dict[str, set]:
    """Aligned-subset filter: dataset -> kept example ids. Canonical test ids
    carry their dataset as prefix; OOD sets live in extra_ids (nq/triviaqa)."""
    s = json.loads(Path(scope_path).read_text())
    keep = {ds: {i for i in s["ids"]["test"] if i.startswith(ds + "-")}
            for ds in ("hotpotqa", "twowiki", "musique", "popqa")}
    keep["nq_ood"] = set(s["extra_ids"]["nq"])
    keep["triviaqa_ood"] = set(s["extra_ids"]["triviaqa"])
    return keep


def eval_dir(d: Path, keep: set | None = None) -> dict | None:
    cands = sorted(d.glob("**/intermediate_data.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    data = json.loads(cands[-1].read_text())
    if keep is not None:
        data = [it for it in data if it.get("id") in keep]
    if not data:
        return None
    agg = Counter()
    for it in data:
        golds = it.get("golden_answers") or []
        pred = (it.get("output") or {}).get("pred") or ""
        for k, v in prf(pred, golds).items():
            agg[k] += v
    n = len(data)
    cs = sorted(d.glob("**/cost_summary.json"), key=lambda p: p.stat().st_mtime)
    cost = json.loads(cs[-1].read_text()) if cs else {}
    gen = cost.get("gen_calls_per_item")
    tf = sorted(d.glob("**/pgir_trace_*.jsonl"), key=lambda p: p.stat().st_mtime)
    tf += sorted(d.glob("**/probingrag_trace.jsonl"), key=lambda p: p.stat().st_mtime)
    if tf:
        tr = [json.loads(l) for l in open(tf[-1])]
        if keep is not None:
            tr = [t for t in tr if t.get("example_id") in keep]
        if tr:
            gen = sum(t["cost"]["gen_calls"] for t in tr) / len(tr)
    return {"n": n, "em": agg["em"] / n, "f1": agg["f1"] / n,
            "precision": agg["precision"] / n, "recall": agg["recall"] / n,
            "gen_per_item": gen, "retr_per_item": cost.get("retr_calls_per_item")}


def build(methods, datasets, dir_patterns, title, scope: dict | None = None):
    if isinstance(dir_patterns, str):
        dir_patterns = [dir_patterns]
    rows = {}
    for m in methods:
        for ds in datasets:
            hits = []
            for pat in dir_patterns:
                if ("m6_" in pat or "m7_" in pat) and m != "pgir":
                    continue  # m6_* / m7_* dirs hold pgir runs only
                hits += [p for p in RUNS.glob(pat.format(m=m, ds=ds)) if p.is_dir()]
            for d in sorted(hits, key=lambda p: p.stat().st_mtime):
                r = eval_dir(d, keep=scope.get(ds) if scope else None)
                if r:
                    rows[(m, ds)] = r  # newest run wins
    md = [f"# {title}", "", "| method | dataset | n | EM | F1 | Prec | Rec | gen/item |",
          "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        for ds in datasets:
            r = rows.get((m, ds))
            if not r:
                continue
            md.append(f"| {m} | {ds} | {r['n']} | {r['em']:.3f} | {r['f1']:.3f} | "
                      f"{r['precision']:.3f} | {r['recall']:.3f} | "
                      f"{r['gen_per_item'] and round(r['gen_per_item'], 2) or '-'} |")
        # macro row over datasets this method covers fully
        have = [rows[(m, ds)] for ds in datasets if (m, ds) in rows]
        if len(have) == len(datasets) and len(datasets) > 1:
            md.append(f"| {m} | **MACRO** | — | "
                      f"{sum(r['em'] for r in have)/len(have):.3f} | "
                      f"{sum(r['f1'] for r in have)/len(have):.3f} | "
                      f"{sum(r['precision'] for r in have)/len(have):.3f} | "
                      f"{sum(r['recall'] for r in have)/len(have):.3f} | "
                      f"{sum((r['gen_per_item'] or 0) for r in have)/len(have):.2f} |")
    return rows, md


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default=None,
                    help="scope-ids json (e.g. beatrice configs/sampling/fixed_scope_ids.json): "
                         "re-slice metrics to its test/extra ids instead of the full runs")
    ap.add_argument("--suffix", default=None,
                    help="output filename suffix (default '_scoped' when --scope is given)")
    ap.add_argument("--methods", default=None,
                    help="comma list restricting CORE methods (e.g. label-clean "
                         "'norag,vanilla,dragin' while probe/label-dependent systems retrain)")
    args = ap.parse_args()
    scope = load_scope(args.scope) if args.scope else None
    suffix = args.suffix if args.suffix is not None else ("_scoped" if scope else "")
    core = [m for m in CORE_METHODS if m in args.methods.split(",")] \
        if args.methods else CORE_METHODS

    OUT.mkdir(parents=True, exist_ok=True)
    all_out = {}
    main_patterns = ["smoke_{m}_{ds}_test_nfull", "smoke_{m}_{ds}_test_full",
                     "m6_main/{ds}_*", "m6_pure/{ds}_*",
                     "m7_fused/{ds}_*"]  # m6_*/m7_* hold pgir only; newest wins
    # NOTE: the seakr rows in main_full_metrics.md are hand-patched from the
    # faithful official-repo reproduction (data lives in SeaKR/, not under
    # runs/flashrag) — re-apply scripts/patch_seakr_rows after a rebuild.
    scope_note = (f" — scoped subset ({Path(args.scope).name})" if scope else "")
    tables = {
        "main": (core, MAIN_DATASETS, main_patterns,
                 "Main table — confidence-control family; canonical test x4 "
                 "+ user OOD x2; MACRO = 6-set mean" + scope_note),
        "generalization": (core, GEN_DATASETS, "smoke_{m}_{ds}_test_n500*",
                           "LEGACY generalization (native splits, n=500) — superseded by the OOD columns in main"),
        "paradigm_ref": (PARADIGM_REF_METHODS, MAIN_DATASETS,
                         ["smoke_{m}_{ds}_test_nfull", "smoke_{m}_{ds}_test_full"],
                         "Paradigm-reference systems (NOT in main comparison; "
                         "different retrieval/answer paradigms — see RESULTS.md)" + scope_note),
        "paradigm_ref_gen": (PARADIGM_REF_METHODS, GEN_DATASETS,
                             "smoke_{m}_{ds}_test_*",
                             "Paradigm-reference systems, legacy native-split sets"),
        "s3": (["pgir"], ["hotpotqa", "musique"], "s3_ablation_{ds}_dev",
               "S3 ablation (dev n=200)"),
    }
    if scope:  # scoped re-slice: only the test-scope tables make sense
        tables = {k: v for k, v in tables.items() if k in ("main", "paradigm_ref")}
    for name, (methods, datasets, pattern, title) in tables.items():
        rows, md = build(methods, datasets, pattern, title, scope=scope)
        (OUT / f"{name}_full_metrics{suffix}.md").write_text("\n".join(md) + "\n")
        all_out[name] = {f"{m}|{d}": r for (m, d), r in rows.items()}
        print("\n".join(md))
        print()
    (OUT / f"full_metrics{suffix}.json").write_text(json.dumps(all_out, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
