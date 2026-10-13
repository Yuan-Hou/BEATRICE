"""ReDeEP (ICLR'25, Sun et al., 2410.11414) feature extraction for Qwen3.5-9B.

Faithful re-implementation of the official `Jeryi-Sun/ReDEeP-ICLR` scoring, ported
off its vendored-transformers LLaMA patch onto our hybrid-attention Qwen via forward
hooks. ReDeEP decouples a RAG answer into two mechanistic signals:

  * Parametric-Knowledge Score (PKS) -- "Knowledge FFNs". Per decoder layer, the JS
    divergence between the logit-lens of the post-FFN residual and the pre-FFN
    residual (i.e. how much THIS layer's FFN moves the next-token distribution).
    Replicates the patch's `logits_dict[layer] = (lm_head(norm(h_{l+1})),
    lm_head(norm(residual_l)))` with `residual_l = h_{l+1} - mlp_out_l` captured by a
    hook on each layer's `.mlp`. Fully faithful across all 32 layers.

  * External-Context Score (ECS) -- "Copying Heads". Per attention head, the cosine
    similarity between the generated token's last-layer hidden and the mean last-layer
    hidden of the top-10% prefix tokens that head attends to. Qwen3.5-9B only emits
    standard attention on the 8 full-attention layers (indices 3,7,...,31; interval 4),
    so the copy-head search is restricted to those 8*16=128 head candidates. This is a
    forced, documented deviation from the paper (which searches all layers).

Per group we save the per-layer PKS [N,32] and per-head ECS [N,128], plus an index.
Selection of top-K layers / top-N heads / alpha is done downstream (redeep.py),
faithful to `token_level_reg.py`.

Run (vllm2 env; avoid GPUs 0,1 = the user's vLLM server):
    CUDA_VISIBLE_DEVICES=4 .../vllm2/bin/python scripts/extract_redeep.py \
        --config configs/experiment/probe/baselines/baselines_context_withctx_qwen9b.yaml \
        --group validation --shard 0/1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ragprobe.features.extractor import (
    _apply_chat_template, _build_prompt, _find_text_span, _format_context_legacy,
    _get_example,
)
from ragprobe.training.dataset import get_binary_label, load_feature_meta
from ragprobe.utils.logging import log

# Qwen3.5-9B: full-attention layers (the rest are linear-attention, no attn map).
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]          # qwen3.5-9b (16 heads)
GEMMA_FULL_ATTN_LAYERS = [5, 11, 17, 23, 29, 35, 41]        # gemma-4-E4B (8 heads)


def _install_attn_taps(layers, layer_ids, captured):
    """Return eager attention weights from selected layers only (gemma materializes
    all 43 layers under output_attentions=True -> tens of GB on long contexts)."""
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
    runs = record.get("runs", {})
    for key in ("with_context", "without_context"):
        r = runs.get(key) if isinstance(runs, dict) else None
        if isinstance(r, dict) and r.get("extracted_answer"):
            return str(r["extracted_answer"])
    return None


def _js_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Per-row Jensen-Shannon divergence (mean over vocab), matching the official
    `calculate_dist`: 0.5*(KL(P||M)+KL(Q||M)) with mean reduction, scaled by 1e6.
    p_logits/q_logits: [n, V]. Returns [n]."""
    sp = F.softmax(p_logits, dim=-1)
    sq = F.softmax(q_logits, dim=-1)
    m = 0.5 * (sp + sq)
    lsp = F.log_softmax(p_logits, dim=-1)
    lsq = F.log_softmax(q_logits, dim=-1)
    kl1 = F.kl_div(lsp, m, reduction="none").mean(-1)
    kl2 = F.kl_div(lsq, m, reduction="none").mean(-1)
    return 0.5 * (kl1 + kl2) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--group", required=True,
                    help="validation | test_id | popqa_ood (withctx eval groups)")
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-answer-tokens", type=int, default=32)
    ap.add_argument("--top-rate", type=float, default=0.1, help="ECS top attended-token fraction.")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="data/redeep_features/withctx")
    args = ap.parse_args()

    from ragprobe.utils.config import load_yaml
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    probe_cfg = cfg.get("probe", {})
    label_mode = probe_cfg.get("label_mode", "with_context_correct")
    positive_transition = probe_cfg.get("positive_transition", "NY")

    # ---- resolve meta for this group (same loader/scope as everything else) ----
    if args.group == "validation":
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
    log(f"group={args.group} shard={args.shard} -> {len(meta)} records")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    log(f"loading {args.model_name} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, attn_implementation="eager", dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    lm = model.model.language_model
    norm, lm_head = lm.norm, model.lm_head
    n_layers = len(lm.layers)

    # postnorm E-series (gemma): the FFN contribution entering the residual stream is
    # post_feedforward_layernorm(mlp-chain), and the whole layer output is scaled by
    # layer_scalar — hook the norm and remove scalar*capture. prenorm (qwen/llama):
    # raw mlp output IS the contribution (scalar 1).
    is_postnorm = getattr(lm.layers[0], "post_feedforward_layernorm", None) is not None
    ffn_scalars = [float(getattr(lm.layers[i], "layer_scalar", 1.0)) for i in range(n_layers)]

    mlp_out: dict[int, torch.Tensor] = {}

    def _mk_hook(idx):
        def hook(_mod, _inp, out):
            mlp_out[idx] = out if isinstance(out, torch.Tensor) else out[0]
        return hook

    handles = []
    for i in range(n_layers):
        tgt = lm.layers[i].post_feedforward_layernorm if is_postnorm else lm.layers[i].mlp
        handles.append(tgt.register_forward_hook(_mk_hook(i)))

    is_gemma = "gemma" in args.model_name.lower()
    captured_attn: dict = {}
    if is_gemma:
        attn_layers = GEMMA_FULL_ATTN_LAYERS
        _install_attn_taps(lm.layers, attn_layers, captured_attn)
        n_heads = int(model.config.text_config.num_attention_heads)
    else:
        attn_layers = FULL_ATTN_LAYERS
        n_heads = 16
    n_cand = len(attn_layers) * n_heads
    head_ids = [(L, h) for L in attn_layers for h in range(n_heads)]
    pks_all, ecs_all, index = [], [], []
    skipped = 0

    for record in tqdm(meta, desc=f"redeep/{args.group}/{args.shard}"):
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
            mlp_out.clear()
            captured_attn.clear()
            inputs = _apply_chat_template(processor, combined, model.device)
            ids = inputs["input_ids"][0]
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True,
                            output_attentions=not is_gemma, use_cache=False)
            q_start, q_end = _find_text_span(processor.tokenizer, ids, question)
            a_start, a_end = _find_text_span(processor.tokenizer, ids, answer_text, min_start=q_end)
            if a_start == 0 or a_end <= a_start:
                skipped += 1
                continue
            a_end = min(a_end, a_start + args.max_answer_tokens)
            a_pos = torch.arange(a_start, a_end, device=ids.device)
            hs = out.hidden_states  # tuple len n_layers+1

            # ---- PKS: per-layer FFN logit-lens JSD over answer tokens ----
            pks = np.zeros(n_layers, dtype=np.float32)
            for L in range(n_layers):
                post = hs[L + 1][0, a_pos]              # post-FFN residual [n_ans, d]
                pre = post - ffn_scalars[L] * mlp_out[L][0, a_pos]  # pre-FFN residual
                lp = lm_head(norm(post)).float()
                lq = lm_head(norm(pre)).float()
                pks[L] = float(_js_div(lp, lq).mean().item())

            # ---- ECS: per copy-head cosine sim to top-attended prefix tokens ----
            last_hidden = hs[-1][0].float()             # [seq, d] (last layer, un-normed)
            prefix_len = a_start                        # attend only over the prefix
            top_k = max(1, int(prefix_len * args.top_rate))
            # stack the full-attn maps -> [n_cand heads, seq, seq]
            if is_gemma:
                if any(li not in captured_attn for li in attn_layers):
                    raise RuntimeError("attention tap returned nothing")
                attn = torch.cat([captured_attn[li][0] for li in attn_layers], dim=0).float()
            else:
                attn = torch.cat([out.attentions[j][0] for j in range(len(out.attentions))], dim=0).float()
            ecs_tok = []                                # [n_ans, 128]
            cur = last_hidden[a_pos]                    # [n_ans, d]
            for ti, seq_i in enumerate(a_pos.tolist()):
                rows = attn[:, seq_i, :prefix_len]      # [128, prefix_len]
                idx_top = torch.topk(rows, top_k, dim=-1).indices  # [n_cand, k]
                sel = last_hidden[idx_top.reshape(-1)].view(n_cand, top_k, -1).mean(1)  # [n_cand, d]
                cos = F.cosine_similarity(sel, cur[ti].unsqueeze(0).expand_as(sel), dim=-1)
                ecs_tok.append(cos)
            ecs = torch.stack(ecs_tok, 0).mean(0).cpu().numpy().astype(np.float32)  # [128]

            pks_all.append(pks)
            ecs_all.append(ecs)
            index.append({
                "example_id": str(ex.get("id", "")), "dataset": ex.get("dataset", ""),
                "split": ex.get("split", ""),
                "label": int(get_binary_label(record, positive_transition, label_mode)),
                "answer_len": int(a_pos.numel()),
            })
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            continue

    for h in handles:
        h.remove()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.group}__shard{si}of{sn}"
    np.savez(out_dir / f"{tag}.npz",
             pks=np.stack(pks_all) if pks_all else np.zeros((0, n_layers)),
             ecs=np.stack(ecs_all) if ecs_all else np.zeros((0, len(head_ids))),
             head_ids=np.array(head_ids, dtype=np.int32))
    (out_dir / f"{tag}.index.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in index) + "\n", encoding="utf-8")
    log(f"wrote {len(index)} examples (skipped {skipped}) -> {out_dir}/{tag}.npz")


if __name__ == "__main__":
    main()
