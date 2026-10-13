"""Bit-exactness gate for the packed feature path (scripts/pack_sidecar_features.py).

For every packed meta file and a matrix of dataset-flag combinations (covering all
branch-encoder configs plus the Phase B ablation hooks), construct the dataset
twice — slow sidecar path vs PROBE_PACKED_ROOT fast path — and require
torch.equal on every element of the output tuple for a random row sample.

Exit 0 + VERIFY_PACKED_OK only if every comparison passes. This must pass before
sidecars are deleted and before any packed-path training starts.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragprobe.training import packed_store  # noqa: E402
from ragprobe.training.dataset import (  # noqa: E402
    ContextQuestionAnswerSidecarDataset,
    QuestionAnswerSidecarDataset,
)
from pack_sidecar_features import META_FILES  # noqa: E402  (scripts/ on path via __file__ dir)

TOK = SimpleNamespace(pad_token_id=151643)
N_ROWS = 16
PACKED_ROOT = ROOT / "data/packed_features"

CTX_COMBOS = [
    {},  # quad-like: everything on
    {"use_attention_mass": False},
    {"use_numeric_features": False, "use_probability_features": False,
     "use_residual_features": False, "use_attention_features": False},  # hidden-only
    {"use_last_hidden": False, "use_residual_features": False},          # prob+attn
    {"use_probability_features": False},                                  # resid branch prefixes
    {"ablate_segments": ("context", "question")},                         # Phase B span variant
    {"ablate_channels": ("_lens",)},                                      # Phase B channel split
]
Q_COMBOS = [
    {},
    {"use_attention_mass": False},
    {"use_numeric_features": False, "use_probability_features": False,
     "use_residual_features": False, "use_attention_features": False},
    {"use_probability_features": False},
    {"ablate_segments": ("answer",)},
]


def build(family, meta_rows, packed: bool, **kw):
    os.environ["PROBE_PACKED_ROOT"] = str(PACKED_ROOT) if packed else ""
    packed_store._INDEX = None if not packed else packed_store._INDEX  # keep cache when on
    cls = ContextQuestionAnswerSidecarDataset if family == "ctx" else QuestionAnswerSidecarDataset
    common = dict(use_attention_mass=True)
    common.update(kw)
    return cls(meta_rows, TOK, **common)


def main():
    global PACKED_ROOT
    import argparse

    # pin the reduction order to the packer's (_init_worker sets 1 thread): multi-
    # thread float reductions reorder sums and give ULP-level diffs on large
    # attention groups, which torch.equal correctly rejects but we don't care about
    torch.set_num_threads(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", action="append", default=None, metavar="FAMILY:PATH",
                    help="verify these meta files instead of the built-in qwen list")
    ap.add_argument("--packed-root", default=None,
                    help="packed root to verify against (default data/packed_features)")
    args = ap.parse_args()
    if args.packed_root:
        PACKED_ROOT = ROOT / args.packed_root
    metas = ([tuple(m.split(":", 1)) for m in args.meta] if args.meta else META_FILES)

    random.seed(0)
    failures = 0
    for family, meta_rel in metas:
        meta_path = ROOT / meta_rel
        pack_dir = PACKED_ROOT / meta_path.parent.name / meta_path.stem
        if not (pack_dir / "index.pt").exists():
            print(f"[SKIP] not packed yet: {meta_rel}", flush=True)
            failures += 1
            continue
        rows = [json.loads(l) for l in open(meta_path)]
        sample = random.sample(range(len(rows)), min(N_ROWS, len(rows)))
        combos = CTX_COMBOS if family == "ctx" else Q_COMBOS
        for ci, kw in enumerate(combos):
            slow = build(family, rows, packed=False, **kw)
            fast = build(family, rows, packed=True, **kw)
            assert fast._packed is not None, f"packed resolve failed: {meta_rel} combo{ci}"
            assert slow._packed is None
            bad = 0
            for idx in sample:
                a, b = slow[idx], fast[idx]
                assert len(a) == len(b)
                for fi, (x, y) in enumerate(zip(a, b)):
                    if isinstance(x, torch.Tensor):
                        ok = (x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y))
                    else:
                        ok = (x == y)
                    if not ok:
                        bad += 1
                        print(f"[FAIL] {meta_path.stem} combo{ci} row{idx} field{fi}: "
                              f"{getattr(x,'dtype',type(x))}{getattr(x,'shape','')} vs "
                              f"{getattr(y,'dtype',type(y))}{getattr(y,'shape','')}", flush=True)
                        break
            status = "ok" if bad == 0 else f"{bad} BAD"
            print(f"[{meta_path.stem}] combo{ci} {kw or 'all-on'}: {status}", flush=True)
            failures += bad
    if failures:
        print(f"VERIFY_PACKED_FAILED ({failures})", flush=True)
        sys.exit(1)
    print("VERIFY_PACKED_OK", flush=True)


if __name__ == "__main__":
    main()
