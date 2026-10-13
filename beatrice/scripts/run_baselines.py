"""Run the Tier-A correctness-detection baselines on already-extracted features.

Reuses the same meta loading / fixed-scope partition / binary labels as the main probes,
so every baseline scores the identical examples. Writes one prediction JSONL per scorer
(standard schema with ``helpful_probability``) plus a combined metrics JSON + text table.

Usage:
    python scripts/run_baselines.py --config configs/experiment/probe/baselines/baselines_question_noctx_qwen9b.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragprobe.baselines.evaluate import format_report_table, metrics_overall_and_by_dataset
from ragprobe.baselines.extract import extract_split_signals
from ragprobe.baselines.scorers import build_scorers, run_scorer
from ragprobe.training.dataset import load_feature_meta
from ragprobe.utils.config import load_yaml
from ragprobe.utils.io import write_jsonl
from ragprobe.utils.logging import log


def _signals_to_records(rows, scores) -> list[dict]:
    records = []
    for sig, score in zip(rows, scores):
        records.append({
            "example_id": sig.example_id,
            "dataset": sig.dataset,
            "split": sig.split,
            "question": sig.question,
            "answers": sig.answers,
            "transition": sig.transition,
            "label": int(sig.label),
            "helpful_probability": float(score),
            "score": float(score),
            "answer_len": sig.answer_len,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", nargs="*", default=None, help="Subset of scorer names to run.")
    parser.add_argument("--limit", type=int, default=None, help="Cap examples per split (smoke test).")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_cfg = config["data"]
    probe_cfg = config.get("probe", {})
    output_cfg = config.get("output", {})

    segment = probe_cfg.get("segment", "answer")
    label_mode = probe_cfg.get("label_mode", "without_context_correct")
    positive_transition = probe_cfg.get("positive_transition", "NY")

    train_meta = load_feature_meta(
        data_cfg["train_meta_paths"],
        splits=data_cfg.get("train_splits"),
        partition=data_cfg.get("train_partition"),
    )
    val_meta = load_feature_meta(
        data_cfg["validation_meta_paths"],
        splits=data_cfg.get("validation_splits"),
        partition=data_cfg.get("validation_partition"),
    )
    if args.limit:
        train_meta = train_meta[:args.limit]
        val_meta = val_meta[:args.limit]

    log("Extracting baseline signals (train)...")
    train_sig = extract_split_signals(train_meta, segment, label_mode, positive_transition)
    log("Extracting baseline signals (validation)...")
    val_sig = extract_split_signals(val_meta, segment, label_mode, positive_transition)

    # Optional extra eval sets (e.g. test / OOD popqa), same loader.
    evalsets = {"validation": val_sig}
    for extra in data_cfg.get("eval_meta_sets", []) or []:
        meta = load_feature_meta(
            extra["meta_paths"], splits=extra.get("splits"), partition=extra.get("partition"),
        )
        if args.limit:
            meta = meta[:args.limit]
        evalsets[extra["name"]] = extract_split_signals(meta, segment, label_mode, positive_transition)

    out_dir = Path(output_cfg.get("output_dir", "outputs/predictions/baselines"))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_cfg.get("prefix", config.get("experiment_name", "baselines"))
    threshold = float(probe_cfg.get("threshold", 0.5))

    scorers = build_scorers()
    if args.only:
        scorers = {k: v for k, v in scorers.items() if k in set(args.only)}

    report = {}
    for name, (kind, payload) in scorers.items():
        try:
            scored = run_scorer(name, kind, payload, train_sig, evalsets)
        except Exception as exc:  # noqa: BLE001
            log(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            continue
        for split_name, scores in scored.items():
            records = _signals_to_records(evalsets[split_name], scores)
            pred_path = out_dir / f"{prefix}__{name}__{split_name}.jsonl"
            write_jsonl(records, str(pred_path))
            report[f"{name} [{split_name}]"] = metrics_overall_and_by_dataset(records, "helpful_probability", threshold)
        log(f"[{name}] done")

    metrics_path = out_dir / f"{prefix}__metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table = format_report_table(report)
    (out_dir / f"{prefix}__summary.txt").write_text(table + "\n", encoding="utf-8")
    log(f"Wrote metrics to {metrics_path}")
    print("\n" + table)


if __name__ == "__main__":
    main()
