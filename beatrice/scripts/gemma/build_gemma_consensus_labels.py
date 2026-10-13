"""Build the unanimous-consensus label subset for the Gemma migration.

Rule (user, 2026-07-13): a run's label counts only when all three judge
prompts (strict_semantic / alias_friendly / decomposed_fact) parsed and agree.
Records failing unanimity on any evaluated run are excluded from the label
set (kept in the raw generations for later supplementation).

Outputs:
  data/gemma4/labels/<name>_labels.jsonl   one row per kept example:
      {id, dataset, split, runs: {<run>: {is_correct, agreement=1.0}}, transition?}
  data/gemma4/labels/summary.json / summary.md   per-dataset counts, unanimity
      rates, accuracy, and comparison against the Qwen3.5-9B labels.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = ROOT / "data" / "generations"
SRC_DIR = ROOT / "data" / "gemma4" / "sources"
OUT_DIR = ROOT / "data" / "gemma4" / "labels"
QWEN_META = {
    "hotpotqa_supporting_only": ROOT / "data/features_judge_consensus/qwen3_5_9b/hotpotqa_supporting_only_features_meta.jsonl",
    "musique": ROOT / "data/features_judge_consensus/qwen3_5_9b/musique_features_meta.jsonl",
    "twowiki_supporting_only": ROOT / "data/features_judge_consensus/qwen3_5_9b/twowiki_supporting_only_features_meta.jsonl",
    "popqa": ROOT / "data/features_judge_consensus/qwen3_5_9b/popqa_features_meta.jsonl",
    "nq": ROOT / "data/features_judge_consensus/qwen3_5_9b/nq_features_meta.jsonl",
    "triviaqa": ROOT / "data/features_judge_consensus/qwen3_5_9b/triviaqa_features_meta.jsonl",
    "rewritten_context": ROOT / "data/features_judge_consensus/qwen3_5_9b_rewritten_context/rewritten_context_cqa_features_meta.jsonl",
}
NAMES = list(QWEN_META.keys())


def unanimous(run: dict) -> tuple[bool, bool | None]:
    """Return (is_unanimous, verdict). Verdict is None when not unanimous."""
    votes = [d.get("parsed") for d in (run.get("judge_vote") or {}).get("variants", [])]
    if len(votes) != 3 or any(v is None for v in votes):
        return False, None
    if len(set(votes)) != 1:
        return False, None
    return True, bool(votes[0])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in NAMES:
        gen_path = GEN_DIR / f"{name}_gemma4e4b_results.jsonl"
        if not gen_path.exists():
            print(f"skip {name}: no generations yet")
            continue
        rows = [json.loads(l) for l in gen_path.open()]
        kept = []
        n_runs = 0
        n_unan = 0
        acc = {}
        for r in rows:
            ex = r.get("example") or {}
            runs_out = {}
            ok = True
            for rn, run in (r.get("runs") or {}).items():
                if "judge_vote" not in run:
                    # rewritten rows may carry a copied qwen closed-book run
                    # without fresh votes; only freshly judged runs count
                    continue
                n_runs += 1
                u, verdict = unanimous(run)
                if u:
                    n_unan += 1
                    runs_out[rn] = {"is_correct": verdict}
                    acc.setdefault(rn, [0, 0])
                    acc[rn][0] += int(verdict)
                    acc[rn][1] += 1
                else:
                    ok = False
            if ok and runs_out:
                rec = {
                    "id": ex.get("id"),
                    "dataset": ex.get("dataset"),
                    "split": ex.get("split"),
                    "runs": runs_out,
                }
                if {"without_context", "with_context"} <= runs_out.keys():
                    rec["transition"] = ("Y" if runs_out["without_context"]["is_correct"] else "N") + (
                        "Y" if runs_out["with_context"]["is_correct"] else "N")
                kept.append(rec)
        out_path = OUT_DIR / f"{name}_labels.jsonl"
        with out_path.open("w") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        src_n = sum(1 for _ in (SRC_DIR / f"{name}.jsonl").open())
        qwen_n = sum(1 for _ in QWEN_META[name].open()) if QWEN_META[name].exists() else None
        summary[name] = {
            "source_examples": src_n,
            "generated": len(rows),
            "kept_unanimous": len(kept),
            "kept_ratio": round(len(kept) / max(len(rows), 1), 4),
            "judged_runs": n_runs,
            "unanimous_runs": n_unan,
            "run_unanimity_rate": round(n_unan / max(n_runs, 1), 4),
            "accuracy": {rn: round(c / max(t, 1), 4) for rn, (c, t) in sorted(acc.items())},
            "qwen9b_reference_rows": qwen_n,
        }
        print(f"{name}: kept {len(kept)}/{len(rows)} "
              f"(run unanimity {summary[name]['run_unanimity_rate']:.3f}) acc={summary[name]['accuracy']}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Gemma-4-E4B label construction — unanimous 3-prompt consensus subset", "",
             "| dataset | source | generated | kept (all runs unanimous) | run unanimity | acc noctx | acc withctx | qwen9b rows |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['source_examples']} | {s['generated']} | {s['kept_unanimous']} ({s['kept_ratio']:.1%}) "
            f"| {s['run_unanimity_rate']:.1%} | {s['accuracy'].get('without_context', '—')} "
            f"| {s['accuracy'].get('with_context', '—')} | {s['qwen9b_reference_rows']} |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nsummary -> {OUT_DIR}/summary.{{json,md}}")


if __name__ == "__main__":
    main()
