import argparse
import json
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


def _transition(no_correct: bool, rag_correct: bool) -> str:
    return ("Y" if no_correct else "N") + ("Y" if rag_correct else "N")


def _audit_key(record: dict[str, Any]) -> tuple[str, str]:
    return (_dataset(record), _example_id(record))


def _agreement_ok(vote: dict[str, Any], min_agreement: float, require_all_parse: bool) -> bool:
    if vote.get("correct") is None:
        return False
    if float(vote.get("agreement") or 0.0) < min_agreement:
        return False
    if require_all_parse and int(vote.get("num_parse_failed") or 0) > 0:
        return False
    return True


def _audit_record_ok(record: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    if cfg.get("require_parse_ok", True) and not bool(record.get("parse_ok", False)):
        return False, "parse_failed"
    min_agreement = float(cfg.get("min_agreement", 1.0))
    require_all_parse = bool(cfg.get("require_all_prompt_parse", True))
    vote = record.get("judge_vote") or {}
    for run in ("without_context", "with_context"):
        if not _agreement_ok(vote.get(run) or {}, min_agreement, require_all_parse):
            return False, f"low_agreement:{run}"
    if bool(cfg.get("require_original_audit_agree", False)) and bool(record.get("label_changed", False)):
        return False, "label_changed"
    if cfg.get("allowed_issue_types"):
        allowed = {str(x) for x in cfg.get("allowed_issue_types") or []}
        if str(record.get("issue_type")) not in allowed:
            return False, "issue_type"
    return True, "ok"


def _load_audits(paths: list[str], cfg: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    stats = {
        "audit_records": 0,
        "audit_kept": 0,
        "audit_dropped_reasons": Counter(),
        "duplicate_keys": 0,
        "kept_by_dataset": Counter(),
        "kept_transition_audit": Counter(),
    }
    for path in paths:
        for record in read_jsonl(path):
            stats["audit_records"] += 1
            key = _audit_key(record)
            if not key[0] or not key[1]:
                stats["audit_dropped_reasons"]["missing_key"] += 1
                continue
            ok, reason = _audit_record_ok(record, cfg)
            if not ok:
                stats["audit_dropped_reasons"][reason] += 1
                continue
            if key in best:
                stats["duplicate_keys"] += 1
                # Prefer exact 1.0 confidence/agreement records, then later records do not matter much.
                old_conf = float(best[key].get("confidence") or 0.0)
                new_conf = float(record.get("confidence") or 0.0)
                if new_conf < old_conf:
                    continue
            best[key] = record
            stats["audit_kept"] += 1
            stats["kept_by_dataset"][key[0]] += 1
            stats["kept_transition_audit"][str(record.get("transition_audit"))] += 1
    return best, stats


def _apply_audit_labels(item: dict[str, Any], audit: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if not bool(cfg.get("apply_audit_labels", True)):
        return item
    out = dict(item)
    runs = dict(out.get("runs") or {})
    no_correct = bool(audit.get("without_context_correct_audit"))
    rag_correct = bool(audit.get("with_context_correct_audit"))
    no_run = dict(runs.get("without_context") or {})
    rag_run = dict(runs.get("with_context") or {})
    no_run["is_correct"] = no_correct
    rag_run["is_correct"] = rag_correct
    runs["without_context"] = no_run
    runs["with_context"] = rag_run
    out["runs"] = runs
    derived = dict(out.get("derived") or {})
    derived["transition_original_before_purification"] = derived.get("transition")
    derived["transition"] = str(audit.get("transition_audit") or _transition(no_correct, rag_correct))
    out["derived"] = derived
    out["judge_consensus"] = {
        "source": "audit_voting",
        "audit_id": audit.get("audit_id"),
        "transition_original": audit.get("transition_original"),
        "transition_audit": audit.get("transition_audit"),
        "label_changed": bool(audit.get("label_changed", False)),
        "confidence": audit.get("confidence"),
        "issue_type": audit.get("issue_type"),
        "without_context_agreement": ((audit.get("judge_vote") or {}).get("without_context") or {}).get("agreement"),
        "with_context_agreement": ((audit.get("judge_vote") or {}).get("with_context") or {}).get("agreement"),
    }
    return out


def _purify_one(input_path: str, output_path: str, audits: dict[tuple[str, str], dict[str, Any]], cfg: dict[str, Any], scope_ids: set[str] | None = None) -> dict[str, Any]:
    kept = []
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "seen": 0,
        "kept": 0,
        "dropped_no_audit": 0,
        "by_split": Counter(),
        "by_dataset": Counter(),
        "transition_before": Counter(),
        "transition_after": Counter(),
    }
    for item in read_jsonl(input_path):
        stats["seen"] += 1
        key = _audit_key(item)
        if scope_ids is not None and key[1] not in scope_ids:
            continue
        audit = audits.get(key)
        if audit is None:
            stats["dropped_no_audit"] += 1
            continue
        before = str((item.get("derived") or {}).get("transition"))
        purified = _apply_audit_labels(item, audit, cfg)
        after = str((purified.get("derived") or {}).get("transition"))
        kept.append(purified)
        stats["kept"] += 1
        stats["by_split"][_split(purified)] += 1
        stats["by_dataset"][_dataset(purified)] += 1
        stats["transition_before"][before] += 1
        stats["transition_after"][after] += 1
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(kept, output_path)
    return stats


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, defaultdict):
        return dict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    return obj


def run(config: dict[str, Any]) -> None:
    audit_cfg = config.get("audit_filter", {})
    audits, audit_stats = _load_audits(config["input"]["audit_paths"], audit_cfg)
    scope_path = scope_path_from_config(config, "output", "input")
    scope_ids = load_scope_ids(scope_path) if scope_path else None
    outputs = []
    for item in config["input"]["meta_paths"]:
        if isinstance(item, str):
            in_path = item
            out_path = str(Path(config["output"]["meta_dir"]) / Path(item).name)
        else:
            in_path = item["input"]
            out_path = item.get("output") or str(Path(config["output"]["meta_dir"]) / Path(in_path).name)
        outputs.append(_purify_one(in_path, out_path, audits, audit_cfg, scope_ids=scope_ids))
    summary = {
        "audit_filter": audit_cfg,
        "audit_stats": audit_stats,
        "outputs": outputs,
    }
    summary_path = Path(config["output"].get("summary_path") or (str(Path(config["output"]["meta_dir"]) + "_summary.json")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    run(config)


if __name__ == "__main__":
    main()
