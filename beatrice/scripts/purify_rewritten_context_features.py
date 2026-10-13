import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from ragprobe.utils.io import read_jsonl, write_jsonl
from ragprobe.utils.scope import load_scope_ids, scope_path_from_config


def _example_id(record: dict[str, Any]) -> str:
    ex = record.get("example") or {}
    return str(record.get("example_id") or ex.get("id") or record.get("id") or "")


def _dataset(record: dict[str, Any]) -> str:
    ex = record.get("example") or {}
    return str(record.get("dataset") or ex.get("dataset") or "")


def _split(record: dict[str, Any]) -> str:
    ex = record.get("example") or {}
    return str(record.get("split") or ex.get("split") or "")


def _key(record: dict[str, Any]) -> tuple[str, str]:
    return (_dataset(record), _example_id(record))


def _transition(no_correct: bool, rag_correct: bool) -> str:
    return ("Y" if no_correct else "N") + ("Y" if rag_correct else "N")


def _vote_ok(vote: dict[str, Any], min_agreement: float, require_all_parse: bool) -> bool:
    if vote.get("correct") is None and vote.get("final") is None:
        return False
    if float(vote.get("agreement") or 0.0) < min_agreement:
        return False
    if require_all_parse and int(vote.get("num_parse_failed") or 0) > 0:
        return False
    return True


def _load_original_audits(paths: list[str], cfg: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter]:
    min_agreement = float(cfg.get("min_agreement", 1.0))
    require_all_parse = bool(cfg.get("require_all_prompt_parse", True))
    require_parse_ok = bool(cfg.get("require_parse_ok", True))
    audits: dict[tuple[str, str], dict[str, Any]] = {}
    reasons: Counter = Counter()
    for path in paths:
        for record in read_jsonl(path):
            key = _key(record)
            if not key[0] or not key[1]:
                reasons["missing_key"] += 1
                continue
            if require_parse_ok and not bool(record.get("parse_ok", False)):
                reasons["parse_failed"] += 1
                continue
            no_vote = ((record.get("judge_vote") or {}).get("without_context") or {})
            if not _vote_ok(no_vote, min_agreement, require_all_parse):
                reasons["without_context_low_agreement"] += 1
                continue
            audits[key] = record
    return audits, reasons


def _rewritten_with_context_ok(record: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    run = ((record.get("runs") or {}).get("with_context") or {})
    if bool(run.get("judge_parse_failed", False)):
        return False, "with_context_parse_failed"
    vote = run.get("judge_vote") or {}
    if not vote:
        return False, "with_context_missing_vote"
    min_agreement = float(cfg.get("min_agreement", 1.0))
    require_all_parse = bool(cfg.get("require_all_prompt_parse", True))
    if not _vote_ok(vote, min_agreement, require_all_parse):
        return False, "with_context_low_agreement"
    return True, "ok"


def _purify_record(record: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    runs = dict(out.get("runs") or {})
    no_run = dict(runs.get("without_context") or {})
    rag_run = dict(runs.get("with_context") or {})
    no_correct = bool(audit.get("without_context_correct_audit"))
    rag_correct = bool(rag_run.get("is_correct"))
    no_run["is_correct"] = no_correct
    runs["without_context"] = no_run
    runs["with_context"] = rag_run
    out["runs"] = runs

    derived = dict(out.get("derived") or {})
    derived["transition_original_before_purification"] = derived.get("transition")
    derived["transition"] = _transition(no_correct, rag_correct)
    derived["rewritten_context_transition"] = derived["transition"]
    out["derived"] = derived

    out["judge_consensus"] = {
        "source": "original_without_context_audit_plus_rewritten_context_vote",
        "audit_id": audit.get("audit_id"),
        "original_transition_audit": audit.get("transition_audit"),
        "without_context_correct_audit": no_correct,
        "with_context_correct_rewritten": rag_correct,
        "without_context_agreement": ((audit.get("judge_vote") or {}).get("without_context") or {}).get("agreement"),
        "with_context_agreement": (rag_run.get("judge_vote") or {}).get("agreement"),
    }
    return out



def _sample_after_purification(records: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return records
    rng = random.Random(int(cfg.get("seed", 20260611)))
    train_limit = int(cfg.get("train_per_dataset", 0) or 0)
    eval_limit = int(cfg.get("eval_per_dataset", 0) or 0)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(_dataset(record), _split(record))].append(record)
    selected: list[dict[str, Any]] = []
    for (dataset, split), items in sorted(groups.items()):
        limit = train_limit if split == "train" else eval_limit
        items = sorted(items, key=_example_id)
        if limit > 0 and len(items) > limit:
            items = sorted(rng.sample(items, limit), key=_example_id)
        selected.extend(items)
    return selected

def run(config: dict[str, Any]) -> None:
    input_cfg = config["input"]
    filter_cfg = config.get("filter", {})
    audits, audit_drop = _load_original_audits(input_cfg["audit_paths"], filter_cfg)
    scope_path = scope_path_from_config(config, "input", "output")
    scope_ids = load_scope_ids(scope_path) if scope_path else None
    kept = []
    stats = {
        "seen": 0,
        "kept": 0,
        "dropped_no_original_audit": 0,
        "dropped_reasons": Counter(),
        "by_dataset": Counter(),
        "by_split": Counter(),
        "transition_after": Counter(),
        "original_audit_drop_reasons": audit_drop,
    }
    for record in read_jsonl(input_cfg["rewritten_meta_path"]):
        stats["seen"] += 1
        record_key = _key(record)
        if scope_ids is not None and record_key[1] not in scope_ids:
            continue
        audit = audits.get(record_key)
        if audit is None:
            stats["dropped_no_original_audit"] += 1
            continue
        ok, reason = _rewritten_with_context_ok(record, filter_cfg)
        if not ok:
            stats["dropped_reasons"][reason] += 1
            continue
        purified = _purify_record(record, audit)
        kept.append(purified)
        stats["kept"] += 1
        stats["by_dataset"][_dataset(purified)] += 1
        stats["by_split"][_split(purified)] += 1
        stats["transition_after"][(purified.get("derived") or {}).get("transition")] += 1

    pre_sample_kept = len(kept)
    kept = _sample_after_purification(kept, config.get("sampling", {}))
    if len(kept) != pre_sample_kept:
        stats["kept_before_sampling"] = pre_sample_kept
        stats["kept_after_sampling"] = len(kept)
        stats["by_dataset_after_sampling"] = Counter(_dataset(x) for x in kept)
        stats["by_split_after_sampling"] = Counter(_split(x) for x in kept)

    output_path = Path(config["output"]["meta_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(kept, output_path)

    def jsonable(obj: Any) -> Any:
        if isinstance(obj, Counter):
            return dict(obj)
        if isinstance(obj, dict):
            return {str(k): jsonable(v) for k, v in obj.items()}
        return obj

    summary = jsonable(stats)
    summary["output_path"] = str(output_path)
    summary_path = Path(config["output"].get("summary_path", output_path.with_suffix(".summary.json")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    run(config)


if __name__ == "__main__":
    main()
