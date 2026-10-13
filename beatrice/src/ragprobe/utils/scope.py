from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def get_record_id(record: dict[str, Any]) -> str:
    if record.get("id") is not None:
        return str(record.get("id"))
    example = record.get("example")
    if isinstance(example, dict) and example.get("id") is not None:
        return str(example.get("id"))
    if record.get("example_id") is not None:
        return str(record.get("example_id"))
    source = record.get("source_meta")
    if isinstance(source, dict):
        return get_record_id(source)
    return ""


def load_scope_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"scope id file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    data = json.loads(text)
    if isinstance(data, list):
        return {str(x) for x in data}
    ids = data.get("ids")
    if ids is not None:
        return {str(x) for x in ids}
    by_dataset_split = data.get("by_dataset_split") or {}
    out: set[str] = set()
    for split_map in by_dataset_split.values():
        if isinstance(split_map, dict):
            for values in split_map.values():
                out.update(str(x) for x in (values or []))
    return out


def filter_records_by_scope(records: Iterable[dict[str, Any]], scope_ids: set[str]) -> list[dict[str, Any]]:
    records = list(records)
    if not scope_ids:
        return records
    return [record for record in records if get_record_id(record) in scope_ids]


def scope_path_from_config(config: dict[str, Any], *sections: str) -> str | None:
    if config.get("scope_ids_path"):
        return config.get("scope_ids_path")
    if config.get("sample_scope", {}).get("ids_path"):
        return config.get("sample_scope", {}).get("ids_path")
    for section in sections:
        value = config.get(section, {}) if section else config
        if isinstance(value, dict):
            if value.get("scope_ids_path"):
                return value.get("scope_ids_path")
            if value.get("sample_scope", {}).get("ids_path"):
                return value.get("sample_scope", {}).get("ids_path")
    return None
