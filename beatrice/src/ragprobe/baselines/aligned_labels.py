"""Optional label-override hook for baseline re-runs.

The cache-driven baseline scorers (bclass / tierc / selfcheckgpt / bayesian-SE /
llmsknow) freeze labels into their index files at cache time. This hook can swap
labels by example_id against an alternative judge verdict file and restrict rows
to an alternative scope. It is inert unless env ALIGNED_BASELINES=1 is set and
is NOT used for any result reported in the paper; without the env var the
scripts are bit-identical to the default path.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

LABEL_FILES = {
    "noctx": "data/alt_judge_labels/qonly_labels.jsonl",
    "withctx": "data/alt_judge_labels/ctx_labels.jsonl",
    "rewritten": "data/alt_judge_labels/rewritten_labels.jsonl",
}
GROUP_MAP = {"train": "train", "train_rewritten": "train",
             "validation": "validation", "test_id": "test", "popqa_ood": "test"}


def enabled() -> bool:
    return os.environ.get("ALIGNED_BASELINES") == "1"


def out_root(default: str = "outputs/predictions") -> str:
    return "outputs/aligned/predictions" if enabled() else default


@lru_cache(maxsize=None)
def label_map(setting: str) -> dict:
    """{example_id -> 0/1} from the unanimous judge re-verdicts."""
    out = {}
    for line in open(ROOT / LABEL_FILES[setting]):
        r = json.loads(line)
        if r.get("unanimous"):
            out[str(r["id"])] = int(bool(r["is_correct"]))
    return out


@lru_cache(maxsize=None)
def _scope():
    return json.load(open(ROOT / "configs/sampling/aligned_scope_ids.json"))


@lru_cache(maxsize=None)
def group_ids(group: str) -> frozenset:
    sc = _scope()
    if group in GROUP_MAP:
        return frozenset(sc["ids"][GROUP_MAP[group]])
    if group in sc.get("extra_ids", {}):
        return frozenset(sc["extra_ids"][group])
    raise KeyError(f"unknown aligned group: {group}")


def override(records: list, setting: str, group: str,
             id_key: str = "example_id", label_key: str = "label") -> list:
    """Filter records to the aligned scope + unanimous labels, swap the label.
    Row order is preserved. Non-dict rows are not supported."""
    if not enabled():
        return records
    lm = label_map(setting)
    ids = group_ids(group)
    out = []
    for r in records:
        eid = str(r[id_key])
        if eid not in ids or eid not in lm:
            continue
        r = dict(r)
        r[label_key] = lm[eid]
        out.append(r)
    return out


def override_mask(example_ids, setting: str, group: str):
    """(keep_mask: list[bool], new_labels: list[int]) for array-shaped data."""
    lm = label_map(setting)
    ids = group_ids(group)
    keep, labels = [], []
    for eid in example_ids:
        eid = str(eid)
        ok = eid in ids and eid in lm
        keep.append(ok)
        labels.append(lm.get(eid, 0))
    return keep, labels


def filter_layerdata(ld, setting: str, group: str):
    """Return a copy of a LayerData-like dataclass restricted to the aligned scope
    with swapped labels. Every ndarray field of length N is masked; scalar/None
    fields pass through. No-op (returns ld) when disabled."""
    import dataclasses
    import numpy as np
    if not enabled():
        return ld
    keep, labels = override_mask([str(x) for x in ld.example_id], setting, group)
    keep = np.asarray(keep, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    n = len(keep)
    changes = {}
    for f in dataclasses.fields(ld):
        v = getattr(ld, f.name)
        if f.name == "label":
            changes["label"] = labels[keep]
        elif isinstance(v, np.ndarray) and v.shape and v.shape[0] == n:
            changes[f.name] = v[keep]
    return dataclasses.replace(ld, **changes)
