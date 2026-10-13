"""Probe-reranked SuRe (plug-in demo): rescore SuRe's OWN candidates with the
probe and compare EM/F1 against SuRe's original validation+pairwise-ranking
pick — no new generation, everything replays from the on-disk run dirs.

SuRe spends ~(n + n(n-1)) LLM calls per item on candidate selection (instance
validation + pairwise ranking); the probe replaces all of that with n cheap
/score calls over (retrieved contexts, question, candidate).

Needs the probe_scorer service up (port 8008). Run:
  cd <package_root> && python -m pgir.analysis.sure_probe_rerank
  # or restrict: ... --datasets hotpotqa popqa --pattern smoke_sure_{ds}_test_nfull
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgir.scorer_client import ScorerClient
from pgir.strategies import render_contexts
from pgir.tuning.trace import em, f1

RUNS = Path(__file__).resolve().parent.parent / "runs" / "flashrag"
DEFAULT_DATASETS = ["hotpotqa", "twowiki", "musique", "popqa",
                    "nq_ood", "triviaqa_ood"]


def load_sure_items(run_dir: Path) -> list[dict]:
    cands = sorted(run_dir.glob("**/intermediate_data.json"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        return []
    items = []
    for it in json.loads(cands[-1].read_text()):
        out = it.get("output") or {}
        cand = [c for c in (out.get("candidates") or []) if str(c).strip()]
        if not cand:
            continue  # SuRe itself predicted "" here; nothing to rerank
        items.append({
            "example_id": it["id"], "question": it["question"],
            "golden_answers": it.get("golden_answers") or [],
            "candidates": cand, "sure_pred": out.get("pred") or "",
            "contexts": render_contexts(out.get("retrieval_result") or []),
        })
    return items


def rerank_dataset(ds: str, pattern: str, scorer: ScorerClient) -> dict | None:
    hits = [p for p in RUNS.glob(pattern.format(ds=ds)) if p.is_dir()]
    items = []
    for d in sorted(hits, key=lambda p: p.stat().st_mtime):
        got = load_sure_items(d)
        if got:
            items = got  # newest run wins
    if not items:
        return None

    jobs, owner = [], []
    for i, it in enumerate(items):
        for c in it["candidates"]:
            jobs.append({"question": it["question"], "contexts": it["contexts"],
                         "answer": c, "example_id": it["example_id"]})
            owner.append(i)
    results = scorer.score_many(jobs)

    best: dict[int, tuple[float, str]] = {}
    n_failed = 0
    for k, (i, res) in enumerate(zip(owner, results)):
        if res.get("status") != "success":
            n_failed += 1
            continue
        p = float(res["p_correct"])
        if i not in best or p > best[i][0]:
            best[i] = (p, jobs[k]["answer"])
    # items whose every candidate failed to score fall back to the SuRe pick
    rows = []
    for i, it in enumerate(items):
        probe_pick = best[i][1] if i in best else it["sure_pred"]
        rows.append({
            "example_id": it["example_id"],
            "sure_pred": it["sure_pred"], "probe_pick": probe_pick,
            "sure_em": em(it["sure_pred"], it["golden_answers"]),
            "probe_em": em(probe_pick, it["golden_answers"]),
            "sure_f1": f1(it["sure_pred"], it["golden_answers"]),
            "probe_f1": f1(probe_pick, it["golden_answers"]),
            "n_candidates": len(it["candidates"]),
        })
    n = len(rows)
    summary = {
        "dataset": ds, "n": n, "n_score_failed": n_failed,
        "sure_em": sum(r["sure_em"] for r in rows) / n,
        "probe_em": sum(r["probe_em"] for r in rows) / n,
        "sure_f1": sum(r["sure_f1"] for r in rows) / n,
        "probe_f1": sum(r["probe_f1"] for r in rows) / n,
        "flips_gained": sum(1 for r in rows if r["probe_em"] > r["sure_em"]),
        "flips_lost": sum(1 for r in rows if r["probe_em"] < r["sure_em"]),
        "score_calls_per_item": sum(r["n_candidates"] for r in rows) / n,
    }
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--pattern", default="smoke_sure_{ds}_test_*")
    ap.add_argument("--out", default=str(RUNS.parent / "tables" / "sure_probe_rerank"))
    a = ap.parse_args()

    scorer = ScorerClient()  # raises if the service is down — by design
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = ["# Probe-reranked SuRe vs original SuRe pick", "",
          "| dataset | n | SuRe EM | probe EM | SuRe F1 | probe F1 | +flips | -flips | score/item |",
          "|---|---|---|---|---|---|---|---|---|"]
    all_out = {}
    for ds in a.datasets:
        r = rerank_dataset(ds, a.pattern, scorer)
        if r is None:
            print(f"[skip] {ds}: no sure run found for pattern {a.pattern}")
            continue
        s = r["summary"]
        all_out[ds] = r
        md.append(f"| {ds} | {s['n']} | {s['sure_em']:.3f} | {s['probe_em']:.3f} "
                  f"| {s['sure_f1']:.3f} | {s['probe_f1']:.3f} "
                  f"| {s['flips_gained']} | {s['flips_lost']} "
                  f"| {s['score_calls_per_item']:.1f} |")
        print(json.dumps(s, indent=2))
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    (out_dir / "details.json").write_text(json.dumps(
        {ds: r["summary"] for ds, r in all_out.items()}, indent=2))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
