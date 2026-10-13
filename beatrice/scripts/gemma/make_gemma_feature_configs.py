"""Generate Gemma-4-E4B training-feature extraction configs + pipeline manifest.

Mirrors the qwen L17_rd single-pass recipe (hidden_layer + add_resid_lens +
full-attention maps) on the finalized Gemma data. Big units are sharded by
splitting the input results jsonl so the rolling driver
(scripts/gemma/run_gemma_feature_pipeline.sh) can extract -> pack -> verify ->
delete sidecars per shard, keeping transient disk small.

Usage:
  python scripts/gemma/make_gemma_feature_configs.py --layer 21 [--smoke]
Writes:
  configs/experiment/features/gemma4/{ctx,q}/*.yaml
  configs/experiment/features/gemma4/manifest.json   (or manifest_smoke.json)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

MODEL = "google/gemma-4-E4B-it"
FULL_ATTN_LAYERS = [5, 11, 17, 23, 29, 35, 41]  # gemma-4 full-attention blocks (same as base)
GEN_DIR = "data/gemma4/generations_final"

# name -> (ctx shards, q shards); rewritten_context is ctx-only (augmentation set)
DATASETS = {
    "musique": (2, 1),
    "hotpotqa_supporting_only": (2, 1),
    "twowiki_supporting_only": (2, 1),
    "popqa": (1, 1),
    "nq": (1, 1),
    "triviaqa": (1, 1),
    "rewritten_context": (4, 0),
}


def shard_inputs(name: str, n: int) -> list[str]:
    src = ROOT / GEN_DIR / f"{name}_results.jsonl"
    if n == 1:
        return [str(src.relative_to(ROOT))]
    shard_dir = ROOT / GEN_DIR / "shards"
    shard_dir.mkdir(exist_ok=True)
    lines = src.read_text().splitlines(keepends=True)
    outs = []
    for i in range(n):
        p = shard_dir / f"{name}_results.shard{i}of{n}.jsonl"
        if not p.exists() or sum(1 for _ in p.open()) != len(lines[i::n]):
            p.write_text("".join(lines[i::n]))
        outs.append(str(p.relative_to(ROOT)))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="12-row units into separate _smoke dirs for an end-to-end dry run")
    args = ap.parse_args()

    sfx = "_smoke" if args.smoke else ""
    feat_root = f"data/gemma4/features{sfx}"
    cfg_dir = ROOT / "configs/experiment/features/gemma4"
    units = []

    for family, tt in (("ctx", "context"), ("q", "question")):
        (cfg_dir / family).mkdir(parents=True, exist_ok=True)
        fam_dir = (f"gemma4_L{args.layer}_rd" if family == "ctx"
                   else f"gemma4_question_L{args.layer}_rd")
        for name, (n_ctx, n_q) in DATASETS.items():
            n = n_ctx if family == "ctx" else n_q
            if n == 0:
                continue
            inputs = shard_inputs(name, n) if not args.smoke else shard_inputs(name, 1)[:1]
            if args.smoke and name != "musique":
                continue
            for i, inp in enumerate(inputs):
                tag = name if len(inputs) == 1 else f"{name}_shard{i}of{len(inputs)}"
                qtag = f"{tag}_question" if family == "q" else tag
                base = f"{feat_root}/{fam_dir}/{qtag}"
                cfg = {
                    "experiment_name": f"gemma4_{family}_{tag}{sfx}",
                    "model_name": MODEL,
                    "max_examples": 12 if args.smoke else None,
                    "resume": True,
                    "retry_failed": True,
                    "save_every_batches": 500,
                    "input_result_path": inp,
                    "output": {
                        "meta_path": f"{base}_features_meta.jsonl",
                        "tensor_dir": f"{base}_tensors",
                        "token_ids_dir": f"{base}_token_ids",
                        "sidecar_dir": f"{base}_tensors_sidecars",
                    },
                    "feature": {
                        "target_type": tt,
                        "max_context_len": 2048,
                        "answer_tokens": 32,
                        "prob_top_k": 20,
                        "attn_implementation": "eager",
                        "torch_dtype": "bfloat16",
                        "hidden_layer": args.layer,
                        "add_resid_lens": True,
                        "attention_layers": FULL_ATTN_LAYERS,
                        # feature-answer protocol (user 2026-07-16): teacher-force the
                        # vLLM pre-generated TRAINING-format greedy answers — no HF
                        # generation anywhere, full row coverage.
                        "answer_source": "file",
                        "answer_file": f"data/gemma4/trainfmt_answers/{name}.jsonl",
                    },
                }
                cfg_path = cfg_dir / family / f"{qtag}{sfx}.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                units.append({
                    "family": family,
                    "unit": f"{family}_{qtag}{sfx}",
                    "config": str(cfg_path.relative_to(ROOT)),
                    "meta": f"{base}_features_meta.jsonl",
                    "tensor_dir": f"{base}_tensors",
                    "sidecar_dir": f"{base}_tensors_sidecars",
                    "rows": 12 if args.smoke else sum(1 for _ in (ROOT / inp).open()),
                })

    units.sort(key=lambda u: -u["rows"])  # heaviest first for greedy lane balance
    man = cfg_dir / f"manifest{sfx}.json"
    man.write_text(json.dumps({
        "layer": args.layer,
        "packed_root": f"data/gemma4/packed_features{sfx}",
        "units": units,
    }, indent=1))
    print(f"wrote {len(units)} unit configs; manifest -> {man}")


if __name__ == "__main__":
    main()
