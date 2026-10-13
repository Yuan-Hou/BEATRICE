"""Faithful re-extraction for paper-grade baselines.

The original feature pipeline computed all layers' hidden states + attentions in a
single forward pass but stored ONLY the final layer (`last_hidden_states`). To run
the baselines as their papers specify we need:
  - SAPLMA (Azaria & Mitchell 2023): MLP on the hidden of a *single token*, with a
    *layer sweep* (mid layers usually best).
  - SEP (Kossen 2024): logistic probe on a single token's hidden at a swept layer.
  - LLM-Check (Sriramanan 2024): per-layer hidden log-det (eigen) score + per-layer
    attention log-det score + logit perplexity/entropy, all from one forward pass.

This script re-runs the EXACT same prompt + stored greedy answer (teacher-forced from
the existing sidecar's `answer_text`, so the answer span matches the stored features
1:1) and saves, per split, compact arrays:
    hidden_last  [N, L+1, d]   last answer-token hidden, every layer   (fp16)
    hidden_mean  [N, L+1, d]   mean over answer tokens, every layer    (fp16)
    hid_logdet   [N, L+1]      LLM-Check hidden log-det per layer       (fp32)
    attn_logdet  [N, A]        LLM-Check attention log-det per attn layer (fp32, --with-attention)
    perplexity   [N]           exp(mean surprisal) over answer tokens
    mean_entropy [N], max_entropy [N]
plus index rows {example_id, dataset, split, label, answer_len}.

Run in the vllm2 env (transformers recognizes qwen3_5):
    CUDA_VISIBLE_DEVICES=4 .../vllm2/bin/python scripts/extract_layer_features.py \
        --config configs/experiment/probe/baselines/baselines_question_noctx_qwen9b.yaml \
        --setting noctx --group train --shard 0/3 --with-attention 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ragprobe.features.extractor import (
    _apply_chat_template, _build_prompt, _find_text_span, _format_context_legacy,
    _get_example,
)
from ragprobe.training.dataset import get_binary_label, load_feature_meta
from ragprobe.utils.config import load_yaml
from ragprobe.utils.logging import log


def _stored_answer_text(record: dict) -> str | None:
    """Teacher-force the exact answer the stored features were built on."""
    try:
        sc_path = record["feature_extraction"]["rag"].get("sidecar_path")
    except Exception:
        sc_path = None
    if sc_path and Path(sc_path).exists():
        try:
            sc = torch.load(sc_path, map_location="cpu", weights_only=False)
            at = sc.get("answer_text")
            if at:
                return str(at)
        except Exception:
            pass
    # Fallback: the judged run's extracted answer.
    runs = record.get("runs", {})
    for key in ("without_context", "with_context"):
        r = runs.get(key) if isinstance(runs, dict) else None
        if isinstance(r, dict) and r.get("extracted_answer"):
            return str(r["extracted_answer"])
    return None


def _gram_logdet(h: torch.Tensor) -> float:
    """LLM-Check hidden score: log|det| of the centered token Gram matrix / n.

    h: [n_tokens, d]. Gram is n x n (well-conditioned since d >> n). Matches the
    single-response EigenScore/INSIDE-style covariance log-det used by LLM-Check.
    """
    n = h.shape[0]
    if n < 2:
        return 0.0
    h = h.float()
    h = h - h.mean(dim=0, keepdim=True)
    g = h @ h.t()
    g = g + 1e-3 * torch.eye(n, device=g.device, dtype=g.dtype)
    sign, logabsdet = torch.linalg.slogdet(g)
    return float(logabsdet.item() / n)


def _attn_logdet(attn_qk: torch.Tensor) -> float:
    """LLM-Check attention score: mean over heads of log|det| of the answer-token
    attention sub-matrix. attn_qk: [heads, n, n]."""
    heads, n, _ = attn_qk.shape
    if n < 2:
        return 0.0
    a = attn_qk.float()
    a = a + 1e-3 * torch.eye(n, device=a.device, dtype=a.dtype).unsqueeze(0)
    sign, logabsdet = torch.linalg.slogdet(a)
    return float((logabsdet / n).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--group", required=True,
                    help="train | validation | test_id | popqa_ood | <custom when --meta-paths given>")
    ap.add_argument("--meta-paths", nargs="*", default=None,
                    help="Override meta paths (e.g. rewritten-context aug). Loaded with the "
                         "config's train splits+partition and written under --group's name.")
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-answer-tokens", type=int, default=32)
    ap.add_argument("--with-attention", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n shard of the group.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    probe_cfg = cfg.get("probe", {})
    label_mode = probe_cfg.get("label_mode", "without_context_correct")
    positive_transition = probe_cfg.get("positive_transition", "NY")
    target_type = "question" if args.setting == "noctx" else "context"

    # ---- resolve the meta for this group (same loader/scope as everything else) ----
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
    log(f"group={args.group} setting={args.setting} shard={args.shard} -> {len(meta)} records")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    log(f"loading {args.model_name} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, attn_implementation="eager", torch_dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)

    hid_last, hid_mean, hid_logdet, attn_logdet = [], [], [], []
    ppl, ment, maxent, index = [], [], [], []
    skipped = 0

    for record in tqdm(meta, desc=f"{args.setting}/{args.group}/{args.shard}"):
        ex = _get_example(record)
        question = str(ex.get("question") or "").strip()
        context = None if target_type == "question" else _format_context_legacy(record).strip()
        answer_text = _stored_answer_text(record)
        if not question or (target_type != "question" and not context) or not answer_text:
            skipped += 1
            continue
        _, base_prompt = _build_prompt(question, context)
        combined = base_prompt + " " + answer_text
        try:
            inputs = _apply_chat_template(processor, combined, model.device)
            ids = inputs["input_ids"][0]
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True,
                            output_attentions=bool(args.with_attention))
            # locate answer span (same logic as the original extractor)
            q_start, q_end = _find_text_span(processor.tokenizer, ids, question)
            a_start, a_end = _find_text_span(processor.tokenizer, ids, answer_text, min_start=q_end)
            if a_start == 0 or a_end <= a_start:
                skipped += 1
                continue
            a_end = min(a_end, a_start + args.max_answer_tokens)
            a_pos = torch.arange(a_start, a_end, device=ids.device)
            n_ans = int(a_pos.numel())

            hs = out.hidden_states  # tuple len L+1, each [1, seq, d]
            last_vecs = torch.stack([h[0, a_pos[-1]] for h in hs], 0).float().cpu().numpy().astype(np.float16)
            mean_vecs = torch.stack([h[0, a_pos].mean(0) for h in hs], 0).float().cpu().numpy().astype(np.float16)
            ld = np.array([_gram_logdet(h[0, a_pos]) for h in hs], dtype=np.float32)

            # logit scores over answer tokens (predict pos t from logits at t-1)
            safe = a_pos.clamp_min(1)
            logits = out.logits[0, safe - 1].float()
            probs = torch.softmax(logits, -1)
            tgt = ids[safe].unsqueeze(1)
            p = probs.gather(1, tgt).squeeze(1).clamp_min(1e-10)
            ent = -(probs * (probs + 1e-10).log()).sum(-1)
            surprisal = -p.log()

            hid_last.append(last_vecs); hid_mean.append(mean_vecs); hid_logdet.append(ld)
            ppl.append(float(surprisal.mean().exp().item()))
            ment.append(float(ent.mean().item())); maxent.append(float(ent.max().item()))

            if args.with_attention and out.attentions:
                a_scores = []
                for attn in out.attentions:               # each [1, heads, seq, seq]
                    sub = attn[0][:, a_pos][:, :, a_pos]   # [heads, n_ans, n_ans]
                    a_scores.append(_attn_logdet(sub))
                attn_logdet.append(np.array(a_scores, dtype=np.float32))

            index.append({
                "example_id": str(ex.get("id", "")), "dataset": ex.get("dataset", ""),
                "split": ex.get("split", ""),
                "label": int(get_binary_label(record, positive_transition, label_mode)),
                "answer_len": n_ans,
            })
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            continue

    out_dir = Path(args.out_dir or f"data/layer_features/{args.setting}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.group}__shard{si}of{sn}"
    payload = dict(
        hidden_last=np.stack(hid_last) if hid_last else np.zeros((0,)),
        hidden_mean=np.stack(hid_mean) if hid_mean else np.zeros((0,)),
        hid_logdet=np.stack(hid_logdet) if hid_logdet else np.zeros((0,)),
        perplexity=np.array(ppl, np.float32), mean_entropy=np.array(ment, np.float32),
        max_entropy=np.array(maxent, np.float32),
    )
    if args.with_attention and attn_logdet:
        payload["attn_logdet"] = np.stack(attn_logdet)
    np.savez(out_dir / f"{tag}.npz", **payload)
    (out_dir / f"{tag}.index.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in index) + "\n", encoding="utf-8")
    log(f"wrote {len(index)} examples (skipped {skipped}) -> {out_dir}/{tag}.npz")


if __name__ == "__main__":
    main()
