"""Pack sidecar-derived training features into flat variable-length binaries.

Motivation (user, 2026-07-15): encoder training is CPU-pipeline-bound — per 256-row
batch the GPU computes ~2s while torch.load deserialization (0.8s), entropy/slice
transforms (1.7s) and collate (2.4s) run on one core; observed GPU duty cycle
~0-20%. Everything the datasets emit is deterministic, so we precompute it once,
trimmed to true span lengths (context averages ~208 of the 2048-token cap), and
the runtime read path becomes slice + zero-pad.

For every meta jsonl referenced by the training configs this writes
  data/packed_features/<family>/<meta-stem>/
    index.pt              row index: tensor_path -> row, per-segment lengths,
                          element offsets per field, feature_names, constants
    feats_<s>.f16.bin     [C=137, len]   full numeric-feature slice per segment
    tok_<s>.i32.bin       [len]          token ids per segment
    hid_<s>.f16.bin       [len, D=4096]  L17 hidden states per segment
    attn_<s>.f32.bin      [A, len]       attention entropy+mass groups, the exact
                                         f32 values the dataset transform yields
segments s: c,q,a for the with-context family; q,a for the question-only family.

Bit-exactness contract: values are stored exactly as the slow path computes them
(fp16 slices byte-identical; attention groups computed here with the same code on
the same inputs, stored f32). The runtime pad-to-cap equals the slow path's
pad-to-cap because padding is zeros in both. Verified end-to-end by
scripts/verify_packed_features.py (torch.equal over full output tuples).

Usage:
  python scripts/pack_sidecar_features.py [--workers 12] [--only <substr>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragprobe.training.dataset import (  # noqa: E402
    _attention_entropy,
    _get_sidecar_path,
    _get_tensor_path,
    _get_token_ids_path,
    _packed_span,
    _pad_attn,
)

OUT_ROOT = ROOT / "data/packed_features"
CAPS = {"c": 2048, "q": 128, "a": 32}  # uniform across all training configs (verified)

META_FILES = [
    # with-context family (segments c,q,a)
    ("ctx", "data/features_judge_consensus/qwen3_5_9b_L17_rd/musique_features_meta.jsonl"),
    ("ctx", "data/features_judge_consensus/qwen3_5_9b_L17_rd/hotpotqa_supporting_only_features_meta.jsonl"),
    ("ctx", "data/features_judge_consensus/qwen3_5_9b_L17_rd/twowiki_supporting_only_features_meta.jsonl"),
    ("ctx", "data/features_judge_consensus/qwen3_5_9b_L17_rd/popqa_features_meta.jsonl"),
    ("ctx", "data/features_judge_consensus/qwen3_5_9b_L17_rd/nq_features_meta.jsonl"),
    ("ctx", "data/features_judge_consensus/qwen3_5_9b_L17_rd/triviaqa_features_meta.jsonl"),
    ("ctx", "data/features_judge_consensus/qwen3_5_9b_rewritten_context_L17_rd/rewritten_context_cqa_features_meta.jsonl"),
    # question-only family (segments q,a)
    ("q", "data/features_judge_consensus/qwen3_5_9b_question_L17_rd/musique_question_features_meta.jsonl"),
    ("q", "data/features_judge_consensus/qwen3_5_9b_question_L17_rd/hotpotqa_supporting_only_question_features_meta.jsonl"),
    ("q", "data/features_judge_consensus/qwen3_5_9b_question_L17_rd/twowiki_supporting_only_question_features_meta.jsonl"),
    ("q", "data/features_judge_consensus/qwen3_5_9b_question_L17_rd/popqa_question_features_meta.jsonl"),
    ("q", "data/features_judge_consensus/qwen3_5_9b_question_L17_rd/nq_question_features_meta.jsonl"),
    ("q", "data/features_judge_consensus/qwen3_5_9b_question_L17_rd/triviaqa_question_features_meta.jsonl"),
]


def _seg_slice_trim(tensor: torch.Tensor, span: dict, cap: int) -> torch.Tensor:
    """The pre-padding part of dataset._segment_slice: slice [start:end], trim to cap."""
    start = int(span.get("start", 0) or 0)
    end = int(span.get("end", start) or start)
    part = tensor[..., start:end]
    if part.shape[-1] > cap:
        part = part[..., :cap]
    return part


def _hid_trim(sidecar: dict, name: str, cap: int) -> torch.Tensor:
    hidden = (sidecar.get("last_hidden_states") or {}).get(name)
    if hidden is None or not hasattr(hidden, "numel") or hidden.numel() == 0:
        raise ValueError(f"missing last_hidden_states[{name!r}]")
    hidden = hidden.to(torch.float16)
    if hidden.shape[0] > cap:
        hidden = hidden[:cap]
    return hidden.contiguous()


def pack_row_ctx(item: dict) -> dict:
    features = torch.load(_get_tensor_path(item), map_location="cpu", weights_only=True)
    token_ids = torch.load(_get_token_ids_path(item), map_location="cpu", weights_only=True)
    sidecar = torch.load(_get_sidecar_path(item), map_location="cpu", weights_only=False)
    spans = {s: _packed_span(item, n) for s, n in (("c", "context"), ("q", "question"), ("a", "answer"))}
    lens = {s: min(int(spans[s].get("len", 0) or 0), CAPS[s]) for s in spans}

    out = {}
    for s in ("c", "q", "a"):
        out[f"feats_{s}"] = _seg_slice_trim(features, spans[s], CAPS[s]).to(torch.float16).contiguous().numpy()
        out[f"tok_{s}"] = _seg_slice_trim(token_ids, spans[s], CAPS[s]).to(torch.int32).contiguous().numpy()
        out[f"hid_{s}"] = _hid_trim(sidecar, {"c": "context", "q": "question", "a": "answer"}[s], CAPS[s]).numpy()

    maps = (sidecar.get("attention_maps") or {})
    aq, ac, qc = maps.get("answer_to_question"), maps.get("answer_to_context"), maps.get("question_to_context")
    layer_head = 0
    for v in (aq, ac, qc):
        if v is not None and v.numel() > 0:
            layer_head = int(v.shape[0] * v.shape[1])
            break
    c_len, q_len, a_len = lens["c"], lens["q"], lens["a"]
    LH = layer_head

    def groups_or_zero(m, d2, d3, l2, l3):
        if m is not None and m.numel() > 0 and l2 > 0 and l3 > 0:
            m = m.float()[:, :, :l3, :l2]
            return (_attention_entropy(m, dim=2).reshape(-1, l2),
                    _attention_entropy(m, dim=3).reshape(-1, l3),
                    m.sum(dim=2).reshape(-1, l2),
                    m.sum(dim=3).reshape(-1, l3))
        z2 = torch.zeros((LH, 0), dtype=torch.float32)
        return z2, z2, z2, z2

    if LH == 0:
        for s in ("c", "q", "a"):
            out[f"attn_{s}"] = np.zeros((0, lens[s]), dtype=np.float32)
    else:
        c_from_a, a_to_c, c_from_a_mass, a_to_c_mass = groups_or_zero(ac, "c", "a", c_len, a_len)
        c_from_q, q_to_c, c_from_q_mass, q_to_c_mass = groups_or_zero(qc, "c", "q", c_len, q_len)
        q_from_a, a_to_q, q_from_a_mass, a_to_q_mass = groups_or_zero(aq, "q", "a", q_len, a_len)
        # group order == ContextQuestionAnswerSidecarDataset._attn_groups with mass on
        out["attn_c"] = torch.cat([_pad_attn(g, c_len) for g in
                                   (c_from_a, c_from_q, c_from_a_mass, c_from_q_mass)], dim=0).numpy()
        out["attn_q"] = torch.cat([_pad_attn(g, q_len) for g in
                                   (q_from_a, q_to_c, q_from_a_mass, q_to_c_mass)], dim=0).numpy()
        out["attn_a"] = torch.cat([_pad_attn(g, a_len) for g in
                                   (a_to_q, a_to_c, a_to_q_mass, a_to_c_mass)], dim=0).numpy()

    out["_lens"] = {s: {"span": lens[s],
                        "feat": int(out[f"feats_{s}"].shape[-1]),
                        "hid": int(out[f"hid_{s}"].shape[0])} for s in ("c", "q", "a")}
    out["_tensor_path"] = _get_tensor_path(item)
    return out


def pack_row_q(item: dict) -> dict:
    features = torch.load(_get_tensor_path(item), map_location="cpu", weights_only=True)
    token_ids = torch.load(_get_token_ids_path(item), map_location="cpu", weights_only=True)
    sidecar = torch.load(_get_sidecar_path(item), map_location="cpu", weights_only=False)
    spans = {"q": _packed_span(item, "question"), "a": _packed_span(item, "answer")}
    lens = {s: min(int(spans[s].get("len", 0) or 0), CAPS[s]) for s in spans}

    out = {}
    for s in ("q", "a"):
        out[f"feats_{s}"] = _seg_slice_trim(features, spans[s], CAPS[s]).to(torch.float16).contiguous().numpy()
        out[f"tok_{s}"] = _seg_slice_trim(token_ids, spans[s], CAPS[s]).to(torch.int32).contiguous().numpy()
        out[f"hid_{s}"] = _hid_trim(sidecar, {"q": "question", "a": "answer"}[s], CAPS[s]).numpy()

    q_len, a_len = lens["q"], lens["a"]
    aq = ((sidecar.get("attention_maps") or {}).get("answer_to_question"))
    if aq is None or aq.numel() == 0:
        out["attn_q"] = np.zeros((0, q_len), dtype=np.float32)
        out["attn_a"] = np.zeros((0, a_len), dtype=np.float32)
    else:
        LH = int(aq.shape[0] * aq.shape[1])
        aq = aq.float()[:, :, :a_len, :q_len]
        if q_len > 0 and a_len > 0:
            q_ent = _attention_entropy(aq, dim=2).reshape(-1, q_len)
            a_ent = _attention_entropy(aq, dim=3).reshape(-1, a_len)
            q_mass = aq.sum(dim=2).reshape(-1, q_len)
            a_mass = aq.sum(dim=3).reshape(-1, a_len)
        else:
            q_ent = q_mass = torch.zeros((LH, 0), dtype=torch.float32)
            a_ent = a_mass = torch.zeros((LH, 0), dtype=torch.float32)
        # group order == QuestionAnswerSidecarDataset cat([ent, mass])
        out["attn_q"] = torch.cat([_pad_attn(g, q_len) for g in (q_ent, q_mass)], dim=0).numpy()
        out["attn_a"] = torch.cat([_pad_attn(g, a_len) for g in (a_ent, a_mass)], dim=0).numpy()

    out["_lens"] = {s: {"span": lens[s],
                        "feat": int(out[f"feats_{s}"].shape[-1]),
                        "hid": int(out[f"hid_{s}"].shape[0])} for s in ("q", "a")}
    out["_tensor_path"] = _get_tensor_path(item)
    return out


_WORKER_FAMILY = None


def _init_worker(family: str):
    global _WORKER_FAMILY
    _WORKER_FAMILY = family
    torch.set_num_threads(1)


def _work(item: dict) -> dict:
    return pack_row_ctx(item) if _WORKER_FAMILY == "ctx" else pack_row_q(item)


def pack_meta_file(family: str, meta_rel: str, workers: int, out_root: Path | None = None) -> None:
    meta_path = ROOT / meta_rel
    out_dir = (out_root or OUT_ROOT) / meta_path.parent.name / meta_path.stem
    idx_path = out_dir / "index.pt"
    rows = [json.loads(l) for l in open(meta_path)]
    if idx_path.exists():
        old = torch.load(idx_path, weights_only=False)
        if len(old["tensor_path"]) == len(rows):
            print(f"[skip] {meta_rel} ({len(rows)} rows already packed)", flush=True)
            return
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = ("c", "q", "a") if family == "ctx" else ("q", "a")
    fields = [f"{f}_{s}" for s in segs for f in ("feats", "tok", "hid", "attn")]
    dtypes = {"feats": np.float16, "tok": np.int32, "hid": np.float16, "attn": np.float32}
    files = {f: open(out_dir / f"{f}.{np.dtype(dtypes[f.split('_')[0]]).name}.bin", "wb") for f in fields}
    offsets = {f: [0] for f in fields}
    lens_rec = {s: {"span": [], "feat": [], "hid": []} for s in segs}
    tensor_paths = []
    chan = {}       # channel-dim constants, asserted uniform
    feature_names = None

    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(family,)) as pool:
        for i, out in enumerate(pool.imap(_work, rows, chunksize=8)):
            tensor_paths.append(out["_tensor_path"])
            for s in segs:
                for k in ("span", "feat", "hid"):
                    lens_rec[s][k].append(out["_lens"][s][k])
            for f in fields:
                arr = out[f]
                key = f.split("_")[0]
                exp = {"feats": arr.shape[0] if arr.ndim == 2 else None,
                       "hid": arr.shape[1] if arr.ndim == 2 else None,
                       "attn": arr.shape[0], "tok": None}[key] if key != "tok" else None
                if key != "tok":
                    if f not in chan:
                        chan[f] = exp
                    elif chan[f] != exp:
                        raise RuntimeError(f"{meta_rel}: non-uniform channel dim for {f}: {chan[f]} vs {exp} at row {i}")
                arr = np.ascontiguousarray(arr, dtype=dtypes[key])
                arr.tofile(files[f])
                offsets[f].append(offsets[f][-1] + arr.size)
            if feature_names is None:
                fn = rows[i].get("feature_extraction", {}).get("rag", {}).get("feature_names") or rows[i].get("feature_names")
                feature_names = list(fn) if fn else None
            if (i + 1) % 2000 == 0:
                print(f"  {meta_path.stem}: {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    for f in files.values():
        f.close()
    torch.save({"tensor_path": tensor_paths, "lens": lens_rec, "offsets": offsets,
                "chan": chan, "family": family, "caps": CAPS, "segs": segs,
                "feature_names": feature_names},
               idx_path)
    gb = sum((out_dir / p.name).stat().st_size for p in out_dir.iterdir()) / 1e9
    print(f"[done] {meta_rel}: {len(rows)} rows, {gb:.1f}GB, {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="", help="substring filter on meta path")
    ap.add_argument("--meta", action="append", default=None, metavar="FAMILY:PATH",
                    help="pack this meta file instead of the built-in qwen list; "
                         "family is ctx or q; repeatable (e.g. gemma4)")
    ap.add_argument("--out-root", default=None,
                    help="packed output root (default data/packed_features)")
    args = ap.parse_args()
    if args.meta:
        todo = [tuple(m.split(":", 1)) for m in args.meta]
        bad = [f for f, _ in todo if f not in ("ctx", "q")]
        if bad:
            raise SystemExit(f"--meta family must be ctx or q, got {bad}")
    else:
        todo = [(f, m) for f, m in META_FILES if args.only in m]
    out_root = (ROOT / args.out_root) if args.out_root else None
    for family, meta_rel in todo:
        pack_meta_file(family, meta_rel, args.workers, out_root=out_root)
    print("PACK_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
