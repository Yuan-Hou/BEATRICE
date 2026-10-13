import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: Dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(records, tmp_path)
    tmp_path.replace(path)
