#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ragprobe.utils.io import read_jsonl

DEFAULT_DATASETS = {
    "musique": "data/processed/musique_train_validation.jsonl",
    "hotpotqa": "data/processed/hotpotqa_supporting_only_all_splits.jsonl",
    "twowiki": "data/processed/twowiki_supporting_only_all_splits.jsonl",
    "popqa": "data/processed/popqa_test.jsonl",
}
DEFAULT_AUDIT_PATH = "outputs/audits/generation_label_audit_qwen27b_voting.jsonl"
DEFAULT_FEATURE_META_PATHS = [
    "data/features/qwen3_5_9b/musique_features_meta.jsonl",
    "data/features/qwen3_5_9b/hotpotqa_supporting_only_features_meta.jsonl",
    "data/features/qwen3_5_9b/twowiki_supporting_only_features_meta.jsonl",
    "data/features/qwen3_5_9b/popqa_features_meta.jsonl",
]


def _split(record: dict[str, Any]) -> str:
    return str(record.get("split") or (record.get("example") or {}).get("split") or "unknown")


def _dataset(record: dict[str, Any]) -> str:
    return str(record.get("dataset") or (record.get("example") or {}).get("dataset") or "unknown")


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("example_id") or record.get("id") or (record.get("example") or {}).get("id") or "")


def _expand_variants(dataset: str, ids: list[str]) -> list[str]:
    out = set(ids)
    for rid in ids:
        if dataset == "musique" and rid.startswith("musique-") and not rid.startswith("musique-distractor_only-"):
            out.add(rid.replace("musique-", "musique-distractor_only-", 1))
        elif dataset == "hotpotqa" and rid.startswith("hotpotqa-supporting_only-"):
            out.add(rid.replace("hotpotqa-supporting_only-", "hotpotqa-distractor_only-", 1))
        elif dataset == "twowiki" and rid.startswith("twowiki-supporting_only-"):
            out.add(rid.replace("twowiki-supporting_only-", "twowiki-distractor_only-", 1))
    return sorted(out)


def _agreement_ok(vote: dict[str, Any], min_agreement: float, require_all_parse: bool) -> bool:
    if vote.get("correct") is None:
        return False
    if float(vote.get("agreement") or 0.0) < min_agreement:
        return False
    if require_all_parse and int(vote.get("num_parse_failed") or 0) > 0:
        return False
    return True


def _audit_record_ok(record: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if args.require_parse_ok and not bool(record.get("parse_ok", False)):
        return False, "parse_failed"
    vote = record.get("judge_vote") or {}
    for run in ("without_context", "with_context"):
        if not _agreement_ok(vote.get(run) or {}, args.min_agreement, args.require_all_prompt_parse):
            return False, f"low_agreement:{run}"
    if args.require_original_audit_agree and bool(record.get("label_changed", False)):
        return False, "label_changed"
    return True, "ok"


def _load_groups_from_processed() -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    groups_by_dataset: dict[str, dict[str, list[str]]] = {}
    stats = {"source": "processed", "source_counts": {}}
    for dataset, path_str in DEFAULT_DATASETS.items():
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Missing processed dataset for {dataset}: {path}")
        groups: dict[str, list[str]] = defaultdict(list)
        for record in read_jsonl(path):
            rid = _record_id(record)
            if rid:
                groups[_split(record)].append(rid)
        groups_by_dataset[dataset] = groups
        stats["source_counts"][dataset] = {split: len(set(ids)) for split, ids in sorted(groups.items())}
    return groups_by_dataset, stats


def _load_groups_from_audit(args: argparse.Namespace) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    path = Path(args.audit_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing audit file: {path}")

    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    seen_keys: set[tuple[str, str]] = set()
    dropped = Counter()
    raw_counts: dict[str, Counter] = defaultdict(Counter)
    kept_counts: dict[str, Counter] = defaultdict(Counter)

    for record in read_jsonl(path):
        dataset = _dataset(record)
        split = _split(record)
        rid = _record_id(record)
        if not dataset or not rid:
            dropped["missing_key"] += 1
            continue
        if dataset not in DEFAULT_DATASETS:
            continue
        raw_counts[dataset][split] += 1
        ok, reason = _audit_record_ok(record, args)
        if not ok:
            dropped[reason] += 1
            continue
        key = (dataset, rid)
        if key in seen_keys:
            dropped["duplicate_key"] += 1
            continue
        seen_keys.add(key)
        groups[dataset][split].append(rid)
        kept_counts[dataset][split] += 1

    stats = {
        "source": "audit_consensus",
        "audit_path": str(path),
        "audit_filter": {
            "min_agreement": args.min_agreement,
            "require_parse_ok": args.require_parse_ok,
            "require_all_prompt_parse": args.require_all_prompt_parse,
            "require_original_audit_agree": args.require_original_audit_agree,
        },
        "audit_raw_counts": {k: dict(v) for k, v in raw_counts.items()},
        "source_counts": {k: dict(v) for k, v in kept_counts.items()},
        "audit_dropped_reasons": dict(dropped),
    }
    return groups, stats



def _load_groups_from_features(args: argparse.Namespace) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    audit_groups, audit_stats = _load_groups_from_audit(args)
    audit_ok = {
        (dataset, rid)
        for dataset, split_map in audit_groups.items()
        for ids in split_map.values()
        for rid in ids
    }
    paths = args.feature_meta_paths or DEFAULT_FEATURE_META_PATHS
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    raw_counts: dict[str, Counter] = defaultdict(Counter)
    kept_counts: dict[str, Counter] = defaultdict(Counter)
    dropped = Counter()
    seen: set[tuple[str, str]] = set()
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Missing feature meta path: {path}")
        for record in read_jsonl(path):
            dataset = _dataset(record)
            split = _split(record)
            rid = _record_id(record)
            if not dataset or not rid:
                dropped["missing_key"] += 1
                continue
            raw_counts[dataset][split] += 1
            key = (dataset, rid)
            if key in seen:
                dropped["duplicate_feature_key"] += 1
                continue
            seen.add(key)
            if key not in audit_ok:
                dropped["not_judge_consensus"] += 1
                continue
            groups[dataset][split].append(rid)
            kept_counts[dataset][split] += 1
    stats = {
        "source": "features_after_audit_consensus",
        "feature_meta_paths": [str(x) for x in paths],
        "audit_path": str(args.audit_path),
        "audit_filter": audit_stats.get("audit_filter", {}),
        "feature_raw_counts": {k: dict(v) for k, v in raw_counts.items()},
        "source_counts": {k: dict(v) for k, v in kept_counts.items()},
        "feature_dropped_reasons": dict(dropped),
    }
    return groups, stats

def _choose_source(args: argparse.Namespace) -> str:
    if args.source != "auto":
        return args.source
    return "features" if Path(args.audit_path).exists() else "processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="configs/sampling/fixed_scope_ids.json")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--train-per-dataset", type=int, default=10000)
    parser.add_argument("--eval-per-dataset", type=int, default=1000)
    parser.add_argument("--source", choices=["auto", "audit", "processed", "features"], default="auto")
    parser.add_argument("--audit-path", default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--feature-meta-paths", nargs="*", default=None)
    parser.add_argument("--min-agreement", type=float, default=1.0)
    parser.add_argument("--require-parse-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-all-prompt-parse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-original-audit-agree", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing scope file: {output}. Pass --overwrite to regenerate.")

    source = _choose_source(args)
    if source == "audit":
        groups_by_dataset, stats = _load_groups_from_audit(args)
    elif source == "features":
        groups_by_dataset, stats = _load_groups_from_features(args)
    else:
        groups_by_dataset, stats = _load_groups_from_processed()

    rng = random.Random(args.seed)
    by_dataset_split: dict[str, dict[str, list[str]]] = {}
    sampled_counts: dict[str, dict[str, int]] = {}
    requested_counts: dict[str, dict[str, int]] = {}
    shortfalls: dict[str, dict[str, int]] = {}
    all_ids: set[str] = set()

    for dataset in DEFAULT_DATASETS:
        by_dataset_split[dataset] = {}
        sampled_counts[dataset] = {}
        requested_counts[dataset] = {}
        shortfalls[dataset] = {}
        for split, ids in sorted(groups_by_dataset.get(dataset, {}).items()):
            ids = sorted(set(ids))
            limit = args.train_per_dataset if split == "train" else args.eval_per_dataset
            sample = ids if len(ids) <= limit else sorted(rng.sample(ids, limit))
            expanded = _expand_variants(dataset, sample)
            by_dataset_split[dataset][split] = expanded
            sampled_counts[dataset][split] = len(sample)
            requested_counts[dataset][split] = limit
            shortfalls[dataset][split] = max(0, limit - len(sample))
            all_ids.update(expanded)

    data = {
        "seed": args.seed,
        "source": source,
        "train_per_dataset": args.train_per_dataset,
        "eval_per_dataset": args.eval_per_dataset,
        **stats,
        "requested_counts": requested_counts,
        "sampled_counts": sampled_counts,
        "shortfalls": shortfalls,
        "note": "When source=audit, ids are sampled after judge-consensus purification. ids include corresponding distractor_only ids for datasets that have distractor variants.",
        "by_dataset_split": by_dataset_split,
        "ids": sorted(all_ids),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}")
    print(json.dumps({"source": source, "sampled_counts": sampled_counts, "shortfalls": shortfalls, "total_ids_after_variant_expansion": len(all_ids)}, indent=2))


if __name__ == "__main__":
    main()
