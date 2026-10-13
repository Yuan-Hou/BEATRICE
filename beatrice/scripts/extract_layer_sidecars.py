"""Re-extract the per-token hidden segments at an arbitrary layer L and write new
sidecars, so the fusion's hidden branch can be retrained on a mid layer instead of the
final layer (the original pipeline stored only hidden_states[-1]).

For each record we reconstruct the EXACT source prompt the original extractor used
(_build_prompt(question, context) + ' ' + answer_text -> chat template), assert the
tokenized length equals the stored spans.seq_len (verified 5/5 to match), run one
forward, and slice hidden_states[L] at the stored spans.source_prompt positions for
each segment (question/answer, plus context for withctx). The new sidecar is a copy of
the original with ONLY `last_hidden_states` replaced by the layer-L segments; the
tensor_path / token_ids_path / attention_maps / topk are untouched, so the other
fusion branches (prob/residual/attention) are unaffected. A new meta JSONL is written
with sidecar_path repointed.

    CUDA_VISIBLE_DEVICES=3 .../vllm2/bin/python scripts/extract_layer_sidecars.py \
        --meta data/features_judge_consensus/qwen3_5_9b_question/musique_question_features_meta.jsonl \
        --layer 17 --setting noctx --out-root data/features_judge_consensus/qwen3_5_9b_question_L17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ragprobe.features.extractor import (
    _apply_chat_template, _build_prompt, _format_context_legacy, _get_example,
)
from ragprobe.utils.logging import log

SEGMENTS = {"noctx": ["question", "answer"], "withctx": ["context", "question", "answer"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.meta) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out_root = Path(args.out_root)
    sc_dir = out_root / "sidecars"
    sc_dir.mkdir(parents=True, exist_ok=True)
    log(f"{Path(args.meta).name}: {len(rows)} records -> layer {args.layer}")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    log(f"loading {args.model_name} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, attn_implementation="eager", dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    segs = SEGMENTS[args.setting]

    new_rows, skipped = [], 0
    for record in tqdm(rows, desc=Path(args.meta).stem):
        rag = record.get("feature_extraction", {}).get("rag", {})
        sc_path = rag.get("sidecar_path")
        spans = rag.get("spans", {})
        sp = spans.get("source_prompt", {})
        seq_len = spans.get("seq_len")
        if not sc_path or not Path(sc_path).exists() or not sp or seq_len is None:
            skipped += 1
            continue
        try:
            sidecar = torch.load(sc_path, map_location="cpu", weights_only=False)
            ex = _get_example(record)
            question = str(ex.get("question") or "").strip()
            context = None if args.setting == "noctx" else _format_context_legacy(record).strip()
            answer_text = sidecar.get("answer_text") or ""
            if not question or not answer_text or (args.setting == "withctx" and not context):
                skipped += 1
                continue
            _, base_prompt = _build_prompt(question, context)
            combined = base_prompt + " " + answer_text
            inputs = _apply_chat_template(processor, combined, model.device)
            ids = inputs["input_ids"][0]
            if int(ids.shape[0]) != int(seq_len):
                skipped += 1  # tokenization drift -> keep original (don't corrupt alignment)
                continue
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            hL = out.hidden_states[args.layer][0]  # [seq, d]
            new_lh = {}
            for name in segs:
                s = sp.get(name)
                if not s:
                    continue
                new_lh[name] = hL[s["start"]:s["end"]].detach().cpu().to(torch.float16).contiguous()
            new_sidecar = dict(sidecar)
            new_sidecar["last_hidden_states"] = new_lh
            new_sidecar["hidden_layer"] = args.layer
            new_path = sc_dir / Path(sc_path).name
            torch.save(new_sidecar, new_path)
            new_rec = json.loads(json.dumps(record))  # deep copy
            new_rec["feature_extraction"]["rag"]["sidecar_path"] = str(new_path)
            new_rec["feature_extraction"]["rag"]["hidden_layer"] = args.layer
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
