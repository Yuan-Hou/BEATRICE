"""With-context (rewritten) version of the L17 + residual-direction extraction, in ONE
forward per record instead of two.

Combines scripts/extract_layer_sidecars.py and scripts/extract_residual_direction.py:
reconstruct the exact source prompt (rewritten question + rewritten context; verified
10/10 seq_len match), run once with residual hooks, then
  1. slice hidden_states[L] at spans.source_prompt for context/question/answer and write
     a new sidecar with last_hidden_states replaced (everything else copied), and
  2. compute the logit-lens push per layer/contribution, gather at packed
     source_positions, append 64 `resid_*_lens` channels to the tensor.
The new meta repoints BOTH sidecar_path and tensor_path into --out-root.

    CUDA_VISIBLE_DEVICES=4 .../vllm2/bin/python scripts/extract_withctx_L17rd.py \
        --meta data/features_judge_consensus/qwen3_5_9b_rewritten_context/rewritten_context_cqa_features_meta.jsonl \
        --layer 17 --out-root data/features_judge_consensus/qwen3_5_9b_rewritten_context_L17_rd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ragprobe.features.extractor import (
    _build_prompt, _format_context_legacy, _get_example, _run_with_residual_hooks,
)
from ragprobe.utils.logging import log

SEGMENTS = ["context", "question", "answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.meta) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out_root = Path(args.out_root)
    sc_dir = out_root / "sidecars"
    t_dir = out_root / "tensors"
    sc_dir.mkdir(parents=True, exist_ok=True)
    t_dir.mkdir(parents=True, exist_ok=True)
    log(f"{Path(args.meta).name}: {len(rows)} records -> layer {args.layer} + resid-direction")

    torch.set_grad_enabled(False)  # W_U is a model param; keep projections off the graph
    from transformers import AutoModelForImageTextToText, AutoProcessor
    log(f"loading {args.model_name} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, attn_implementation="eager", dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    lm = model.model.language_model
    norm = lm.norm
    W_U = model.lm_head.weight  # [V, d]
    n_layers = len(lm.layers)

    new_rows, skipped = [], 0
    for record in tqdm(rows, desc=Path(args.meta).stem):
        rag = record.get("feature_extraction", {}).get("rag", {})
        tpath = rag.get("tensor_path")
        scp = rag.get("sidecar_path")
        spans = rag.get("spans", {})
        sp_src = spans.get("source_prompt", {})
        seq_len = spans.get("seq_len")
        if (not tpath or not Path(tpath).exists() or not scp or not Path(scp).exists()
                or not sp_src or seq_len is None):
            skipped += 1
            continue
        try:
            sidecar = torch.load(scp, map_location="cpu", weights_only=False)
            src_pos = sidecar.get("source_positions")
            answer_text = sidecar.get("answer_text") or ""
            ex = _get_example(record)
            question = str(ex.get("question") or "").strip()
            context = _format_context_legacy(record).strip()
            if src_pos is None or not question or not answer_text or not context:
                skipped += 1
                continue
            src_pos = src_pos.long()
            _, base_prompt = _build_prompt(question, context)
            combined = base_prompt + " " + answer_text
            inputs, outputs, residuals = _run_with_residual_hooks(model, processor, combined)
            ids = inputs["input_ids"][0]
            if int(ids.shape[0]) != int(seq_len):
                skipped += 1  # tokenization drift -> skip (don't corrupt alignment)
                continue
            hs = outputs.hidden_states  # tuple len n_layers+1, each [1, seq, d]
            seq = int(ids.shape[0])

            # ---- 1. layer-L hidden segments -> new sidecar ----
            hL = hs[args.layer][0]  # [seq, d]
            new_lh = {}
            for name in SEGMENTS:
                s = sp_src.get(name)
                if not s:
                    continue
                new_lh[name] = hL[s["start"]:s["end"]].detach().cpu().to(torch.float16).contiguous()
            new_sidecar = dict(sidecar)
            new_sidecar["last_hidden_states"] = new_lh
            new_sidecar["hidden_layer"] = args.layer
            new_scp = sc_dir / Path(scp).name
            torch.save(new_sidecar, new_scp)

            # ---- 2. directional (logit-lens) residual channels -> augmented tensor ----
            nxt = torch.empty(seq, dtype=torch.long, device=ids.device)
            nxt[:-1] = ids[1:]
            nxt[-1] = ids[-1]
            U = W_U[nxt].float()  # [seq, d]
            has_next = torch.ones(seq, device=ids.device)
            has_next[-1] = 0.0

            def lens(vec):  # <W_U[next], N(vec)> per position
                return (norm(vec).float() * U).sum(-1)  # [seq]

            new_names, new_channels = [], []
            for l in range(n_layers):
                a = residuals.get(f"resid_attn_l{l:02d}")
                m = residuals.get(f"resid_mlp_l{l:02d}")
                hsl, hsl1 = hs[l][0], hs[l + 1][0]
                if a is not None:
                    a = a[0].to(hsl.device)
                    push = (lens(hsl + a) - lens(hsl)) * has_next  # [seq]
                    new_names.append(f"resid_attn_l{l:02d}_lens")
                    new_channels.append(push)
                if m is not None:
                    m = m[0].to(hsl1.device)
                    push = (lens(hsl1) - lens(hsl1 - m)) * has_next
                    new_names.append(f"resid_mlp_l{l:02d}_lens")
                    new_channels.append(push)
            valid = (src_pos > 0).float().to(new_channels[0].device)
            spos = src_pos.clamp(0, seq - 1).to(new_channels[0].device)
            packed = torch.stack([c[spos] * valid for c in new_channels], 0).half().cpu()

            orig = torch.load(tpath, map_location="cpu", weights_only=True)  # [C, max_len]
            aug = torch.cat([orig.half(), packed], 0)
            new_tpath = t_dir / Path(tpath).name
            torch.save(aug, new_tpath)

            new_rec = json.loads(json.dumps(record))
            r2 = new_rec["feature_extraction"]["rag"]
            r2["sidecar_path"] = str(new_scp)
            r2["hidden_layer"] = args.layer
            r2["tensor_path"] = str(new_tpath)
            r2["feature_names"] = list(rag["feature_names"]) + new_names
            r2["num_feature_channels"] = len(r2["feature_names"])
            new_rows.append(new_rec)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            continue

    out_meta = out_root / Path(args.meta).name
    out_meta.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows) + "\n",
                        encoding="utf-8")
    log(f"wrote {len(new_rows)} records (skipped {skipped}) -> {out_meta}")


if __name__ == "__main__":
    main()
