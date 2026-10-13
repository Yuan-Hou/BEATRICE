"""Lookback Lens (Chuang 2024, EMNLP) feature extraction for Qwen3.5-9B.

Faithful to voidism/Lookback-Lens step01: for each generated (answer) token and each
attention head, the lookback ratio
    LB = mean_attn(context tokens) / (mean_attn(context) + mean_attn(generated-so-far))
measures how much that head looks back at the retrieved context vs the tokens it just
generated. The per-example feature is the lookback ratio averaged over the answer
tokens, one value per (layer, head). A logistic-regression detector is fit downstream
(step03) on these features.

Qwen3.5-9B emits standard attention on only the 8 full-attention layers (L3,7,...,31),
so we get 8*16 = 128 head features (the original uses all layers) -- a documented,
architecture-forced deviation, identical to the ReDeEP ECS limitation.

Run (vllm2 env; avoid GPUs 0,1):
    CUDA_VISIBLE_DEVICES=3 .../vllm2/bin/python scripts/extract_lookback.py --group validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ragprobe.features.extractor import (
    _apply_chat_template, _build_prompt, _find_text_span, _format_context_legacy, _get_example,
)
from ragprobe.training.dataset import get_binary_label, load_feature_meta
from ragprobe.utils.config import load_yaml
from ragprobe.utils.logging import log

FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]          # qwen3.5-9b (16 heads -> 128 feats)
GEMMA_FULL_ATTN_LAYERS = [5, 11, 17, 23, 29, 35, 41]        # gemma-4-E4B (sliding-window layers excluded)


def _install_attn_taps(layers, layer_ids, captured: dict):
    """Force eager attention weights out of selected layers only — gemma returns
    attentions for ALL 43 layers under output_attentions=True, which materializes
    tens of GB on long contexts. Same tap as extract_gemma_layer_features."""
    for li in layer_ids:
        mod = layers[li].self_attn
        orig = mod.forward

        def fwd(*a, _orig=orig, _li=li, **k):
            k["output_attentions"] = True
            out = _orig(*a, **k)
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                captured[_li] = out[1]
            return out

        mod.forward = fwd


def _stored_answer_text(record: dict) -> str | None:
    try:
        sc_path = record["feature_extraction"]["rag"].get("sidecar_path")
    except Exception:
        sc_path = None
    if sc_path and Path(sc_path).exists():
        try:
            sc = torch.load(sc_path, map_location="cpu", weights_only=False)
            if sc.get("answer_text"):
                return str(sc["answer_text"])
        except Exception:
            pass
    runs = record.get("runs", {})
    for key in ("with_context", "without_context"):
        r = runs.get(key) if isinstance(runs, dict) else None
        if isinstance(r, dict) and r.get("extracted_answer"):
            return str(r["extracted_answer"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/probe/baselines/baselines_context_withctx_qwen9b.yaml")
    ap.add_argument("--group", required=True,
                    help="train | validation | test_id | popqa_ood | nq | triviaqa | <custom when --meta-paths given>")
    ap.add_argument("--meta-paths", nargs="*", default=None,
                    help="Override meta paths (e.g. rewritten-context aug). Loaded under the config's "
                         "TRAIN splits+partition and written under --group's name. Mirrors extract_layer_features.py.")
    ap.add_argument("--shard", default="0/1", help="i/n shard of the group (parallel/resumable extraction).")
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-answer-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="data/lookback_features/withctx")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    probe_cfg = cfg.get("probe", {})
    label_mode = probe_cfg.get("label_mode", "with_context_correct")
    positive_transition = probe_cfg.get("positive_transition", "NY")

    if args.meta_paths:
        # Custom meta (e.g. rewritten-context augmentation) under the TRAIN scope/partition.
        meta = load_feature_meta(args.meta_paths, splits=data_cfg.get("train_splits"),
                                 partition=data_cfg.get("train_partition"))
    elif args.group == "train":
        meta = load_feature_meta(data_cfg["train_meta_paths"], splits=data_cfg.get("train_splits"),
                                 partition=data_cfg.get("train_partition"))
    elif args.group == "validation":
        meta = load_feature_meta(data_cfg["validation_meta_paths"], splits=data_cfg.get("validation_splits"),
                                 partition=data_cfg.get("validation_partition"))
    else:
        extra = next(e for e in data_cfg["eval_meta_sets"] if e["name"] == args.group)
        meta = load_feature_meta(extra["meta_paths"], splits=extra.get("splits"), partition=extra.get("partition"))
    meta.sort(key=lambda m: str(_get_example(m).get("id", "")))
    si, sn = (int(x) for x in args.shard.split("/"))
    meta = meta[si::sn]
    if args.limit:
        meta = meta[:args.limit]
    log(f"group={args.group} -> {len(meta)} records")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    log(f"loading {args.model_name} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, attn_implementation="eager", dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)

    is_gemma = "gemma" in args.model_name.lower()
    captured_attn: dict = {}
    if is_gemma:
        attn_layers = GEMMA_FULL_ATTN_LAYERS
        _install_attn_taps(model.model.language_model.layers, attn_layers, captured_attn)
        n_heads = int(model.config.text_config.num_attention_heads)
    else:
        attn_layers = FULL_ATTN_LAYERS
        n_heads = 16

    head_ids = [(L, h) for L in attn_layers for h in range(n_heads)]
    feats, index, skipped = [], [], 0
    for record in tqdm(meta, desc=f"lookback/{args.group}"):
        ex = _get_example(record)
        question = str(ex.get("question") or "").strip()
        context = _format_context_legacy(record).strip()
        answer_text = _stored_answer_text(record)
        if not question or not context or not answer_text:
            skipped += 1
            continue
        _, base_prompt = _build_prompt(question, context)
        combined = base_prompt + " " + answer_text
        try:
            inputs = _apply_chat_template(processor, combined, model.device)
            ids = inputs["input_ids"][0]
            captured_attn.clear()
            with torch.no_grad():
                out = model(**inputs, output_attentions=not is_gemma, use_cache=False)
            q_start, q_end = _find_text_span(processor.tokenizer, ids, question)
            a_start, a_end = _find_text_span(processor.tokenizer, ids, answer_text, min_start=q_end)
            if a_start == 0 or a_end <= a_start:
                skipped += 1
                continue
            a_end = min(a_end, a_start + args.max_answer_tokens)
            # [n_layers*n_heads, seq, seq]
            if is_gemma:
                if any(li not in captured_attn for li in attn_layers):
                    raise RuntimeError("attention tap returned nothing; check transformers version")
                attn = torch.cat([captured_attn[li][0] for li in attn_layers], dim=0).float()
            else:
                attn = torch.cat([out.attentions[j][0] for j in range(len(out.attentions))], dim=0).float()
            lb_per_tok = []
            for seq_i in range(a_start, a_end):
                ctx = attn[:, seq_i, :a_start].mean(-1)                 # [128]
                gen = attn[:, seq_i, a_start:seq_i + 1].mean(-1)        # [128]
                lb_per_tok.append(ctx / (ctx + gen + 1e-8))
            feat = torch.stack(lb_per_tok, 0).mean(0).cpu().numpy().astype(np.float32)  # [128]
            feats.append(feat)
            index.append({
                "example_id": str(ex.get("id", "")), "dataset": ex.get("dataset", ""),
                "split": args.group,
                "label": int(get_binary_label(record, positive_transition, label_mode)),
            })
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            continue

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.group if sn == 1 else f"{args.group}__shard{si}of{sn}"
    np.savez(out_dir / f"{tag}.npz",
             feat=np.stack(feats) if feats else np.zeros((0, len(head_ids))),
             head_ids=np.array(head_ids, dtype=np.int32))
    (out_dir / f"{tag}.index.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in index) + "\n", encoding="utf-8")
    log(f"wrote {len(index)} examples (skipped {skipped}) -> {out_dir}/{tag}.npz")


if __name__ == "__main__":
    main()
