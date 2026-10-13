"""EigenScore / INSIDE (Chen 2024, ICLR) baseline.

Faithful to alibaba/eigenscore `getEigenIndicator`: for the K stochastic samples we
already drew (data/se_samples), embed each sample by the MIDDLE-layer hidden state
mean-pooled over the sample's answer tokens, then

    EigenScore = log10 det( Cov(K embeddings) + alpha * I ),   alpha = 1e-3

(the regularized log-determinant of the K x K covariance of the sample embeddings;
higher = more dispersed answers = more uncertain/hallucination). We teacher-force each
stored sample text after the same prompt the samples were drawn from, so the embedding
matches the sampled answer 1:1. Output scores are oriented higher = more likely CORRECT.

Run (vllm2 env; avoid GPUs 0,1):
    CUDA_VISIBLE_DEVICES=2 .../vllm2/bin/python scripts/extract_eigenscore.py --setting withctx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ragprobe.features.extractor import _apply_chat_template
from ragprobe.utils.logging import log

GROUPS = ["validation", "test_id", "popqa_ood"]
ALPHA = 1e-3


def _eigenscore(emb: np.ndarray) -> float:
    """log10 det(cov(K) + alpha I); emb [K, d]. Matches getEigenIndicator."""
    if emb.shape[0] < 2:
        return 0.0
    cov = np.cov(emb)  # [K, K] covariance across the K samples
    cov = cov + ALPHA * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov)
    return float(logdet / np.log(10.0))  # log10 det


def main():
    global GROUPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out-dir", default="data/eigenscore")
    ap.add_argument("--samples-base", default="data/se_samples", help="sampled-answers root (gemma: data/gemma4/samples)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--groups", nargs="*", default=GROUPS,
                    help="Which eval groups to process (default: %(default)s).")
    args = ap.parse_args()
    GROUPS = list(args.groups)
    setting = args.setting

    from transformers import AutoModelForImageTextToText, AutoProcessor
    log(f"loading {args.model_name} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, attn_implementation="eager", dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    tok = processor.tokenizer

    out_dir = Path(args.out_dir) / setting
    out_dir.mkdir(parents=True, exist_ok=True)

    for g in GROUPS:
        rows = [json.loads(l) for l in open(f"{args.samples_base}/{setting}/{g}.jsonl") if l.strip()]
        if args.limit:
            rows = rows[:args.limit]
        out = []
        for ex in tqdm(rows, desc=f"eigenscore/{setting}/{g}"):
            prompt = ex.get("prompt") or ""
            texts = [(s.get("text") or "").strip() for s in ex["samples"]]
            texts = [t for t in texts if t]
            if not prompt or len(texts) < 2:
                continue
            try:
                prefix = _apply_chat_template(processor, prompt, model.device)
                ids_prefix = prefix["input_ids"][0]
                prefix_len = int(ids_prefix.shape[0])
                # Build the K (prefix + sample) sequences and run ONE padded batch.
                seqs, ans_lens = [], []
                for t in texts:
                    t_ids = tok(t, add_special_tokens=False, return_tensors="pt").input_ids[0].to(model.device)
                    seqs.append(torch.cat([ids_prefix, t_ids]))
                    ans_lens.append(int(t_ids.shape[0]))
                maxlen = max(int(s.shape[0]) for s in seqs)
                pad_id = tok.pad_token_id or tok.eos_token_id
                K = len(seqs)
                batch = torch.full((K, maxlen), pad_id, dtype=seqs[0].dtype, device=model.device)
                attn = torch.zeros((K, maxlen), dtype=torch.long, device=model.device)
                for i, s in enumerate(seqs):
                    batch[i, :s.shape[0]] = s
                    attn[i, :s.shape[0]] = 1
                with torch.no_grad():
                    o = model(input_ids=batch, attention_mask=attn, output_hidden_states=True, use_cache=False)
                mid = len(o.hidden_states) // 2
                hid = o.hidden_states[mid]  # [K, maxlen, d]
                embs = []
                for i in range(K):
                    a0, a1 = prefix_len, prefix_len + ans_lens[i]
                    h = hid[i, a0:a1] if ans_lens[i] > 0 else hid[i, a0 - 1:a0]
                    embs.append(h.float().mean(0).cpu().numpy())
                if len(embs) < 2:
                    continue
                es = _eigenscore(np.stack(embs))
                out.append({"example_id": str(ex["example_id"]), "dataset": ex.get("dataset", ""),
                            "split": g, "label": int(ex["label"]), "eigenscore": es})
            except Exception as exc:  # noqa: BLE001
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                continue
        (out_dir / f"{g}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
        log(f"[{setting}/{g}] wrote {len(out)} examples")


if __name__ == "__main__":
    main()
