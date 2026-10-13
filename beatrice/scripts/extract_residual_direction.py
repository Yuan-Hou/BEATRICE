"""Add DIRECTIONAL residual channels (logit-lens push) to the packed feature tensor.

The residual modality currently keeps only the per-token NORM of each layer's attn/mlp
contribution (magnitude, no direction). This adds, per layer and contribution type, a
ReDeEP-PKS-style directional scalar: how much that layer's update raises the logit of
the REALIZED next token under the model's own unembedding (logit lens):

    resid_{attn,mlp}_l{l}_lens[t] = <W_U[id_{t+1}], N(post_t)> - <W_U[id_{t+1}], N(pre_t)>

where for the mlp at layer l: post = hs[l+1], pre = hs[l+1] - mlp_contrib_l;
for the attn: pre = hs[l], post = hs[l] + attn_contrib_l; N = final RMSNorm; id_{t+1}
the realized next token. Symmetric with the existing *_norm channels (every layer, both
attn & mlp), so the residual branch (prefix `resid_`) auto-includes them.

We re-run the exact source prompt (reconstructed, seq_len asserted == stored), gather at
the stored packed source_positions, append the new channels to the original tensor, and
write an augmented tensor + meta (tensor_path repointed; sidecar_path kept -> so reading
an L17 meta yields L17 hidden + residual-direction together).

    CUDA_VISIBLE_DEVICES=3 .../vllm2/bin/python scripts/extract_residual_direction.py \
        --meta data/.../qwen3_5_9b_question_L17/musique_question_features_meta.jsonl \
        --setting noctx --out-root data/.../qwen3_5_9b_question_L17_rd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ragprobe.features.extractor import (
    _apply_chat_template, _build_prompt, _format_context_legacy, _get_example,
    _run_with_residual_hooks,
)
from ragprobe.utils.logging import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.meta) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out_root = Path(args.out_root)
    t_dir = out_root / "tensors"
    t_dir.mkdir(parents=True, exist_ok=True)
    log(f"{Path(args.meta).name}: {len(rows)} records")

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
        seq_len = spans.get("seq_len")
        if not tpath or not Path(tpath).exists() or not scp or not Path(scp).exists() or seq_len is None:
            skipped += 1
            continue
        try:
            sidecar = torch.load(scp, map_location="cpu", weights_only=False)
            src_pos = sidecar.get("source_positions")
            if src_pos is None:
                skipped += 1
                continue
            src_pos = src_pos.long()
            max_len = int(src_pos.shape[0])
            ex = _get_example(record)
            question = str(ex.get("question") or "").strip()
            context = None if args.setting == "noctx" else _format_context_legacy(record).strip()
            answer_text = sidecar.get("answer_text") or ""
            if not question or not answer_text:
                skipped += 1
                continue
            _, base_prompt = _build_prompt(question, context)
            combined = base_prompt + " " + answer_text
            inputs, outputs, residuals = _run_with_residual_hooks(model, processor, combined)
            ids = inputs["input_ids"][0]
            if int(ids.shape[0]) != int(seq_len):
                skipped += 1
                continue
            hs = outputs.hidden_states  # tuple len n_layers+1, each [1, seq, d]
            seq = int(ids.shape[0])
            # realized next-token unembedding rows: U[t] = W_U[ids[t+1]]
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
            # gather at packed source positions -> [n_new, max_len]
            valid = (src_pos > 0).float().to(new_channels[0].device)
            sp = src_pos.clamp(0, seq - 1).to(new_channels[0].device)
            packed = torch.stack([c[sp] * valid for c in new_channels], 0).half().cpu()  # [n_new, max_len]

            orig = torch.load(tpath, map_location="cpu", weights_only=True)  # [C, max_len]
            aug = torch.cat([orig.half(), packed], 0)
            new_tpath = t_dir / Path(tpath).name
            torch.save(aug, new_tpath)

            new_rec = json.loads(json.dumps(record))
            r2 = new_rec["feature_extraction"]["rag"]
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
