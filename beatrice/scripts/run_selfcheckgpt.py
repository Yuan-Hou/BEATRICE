"""SelfCheckGPT (Manakul 2023) baseline via the official `potsawee/selfcheckgpt`.

Black-box, sampling-consistency hallucination detection: score the main (greedy)
answer against the K stochastic samples we already drew (data/se_samples). We use:
  * SelfCheck-NLI : mean P(contradiction | answer, sample_i) over the K samples,
    with the official DeBERTa-v3-large-MNLI (potsawee/deberta-v3-large-mnli).
  * SelfCheck-Ngram (unigram): surprisal of the answer under a unigram model built
    from {answer + samples}; we report the avg-neg-logprob aggregate.
Because the task is short-answer QA correctness, both sides use the EXTRACTED answers
(greedy vs each sample's extracted answer), matching the SE setup. All written scores
are oriented higher = more likely CORRECT.

    CUDA_VISIBLE_DEVICES=4 .../vllm2/bin/python scripts/run_selfcheckgpt.py --setting withctx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "selfcheckgpt"))
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckNgram  # noqa: E402

GROUPS = ["validation", "test_id", "popqa_ood"]
NLI_MODEL = "potsawee/deberta-v3-large-mnli"


def _read(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    global GROUPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--samples-dir", default=None,
                    help="Override the samples dir (e.g. training-format resamples).")
    ap.add_argument("--groups", nargs="*", default=GROUPS,
                    help="Which eval groups to process (default: %(default)s).")
    args = ap.parse_args()
    GROUPS = list(args.groups)
    setting = args.setting
    from ragprobe.baselines import aligned_labels as AL
    out_dir = Path(args.out_dir or f"{AL.out_root()}/baselines/faithful/{setting}")

    nli = SelfCheckNLI(nli_model=NLI_MODEL, device=torch.device(args.device))
    ngram = SelfCheckNgram(n=1)

    # transformers 5.x removed Tokenizer.batch_encode_plus (used verbatim by the
    # official SelfCheckNLI.predict). Shim it to the modern __call__ so their code
    # runs unchanged; identical tokenization, just the new entrypoint.
    _tok = nli.tokenizer
    if not hasattr(_tok, "batch_encode_plus"):
        def _bep(batch_text_or_text_pairs=None, **kw):
            texts = [p[0] for p in batch_text_or_text_pairs]
            pairs = [p[1] for p in batch_text_or_text_pairs]
            return _tok(text=texts, text_pair=pairs, **kw)
        _tok.batch_encode_plus = _bep

    sdir = args.samples_dir or f"data/se_samples/{setting}"
    summary = []
    for g in GROUPS:
        src = f"{sdir}/{g}.jsonl"
        recs_nli, recs_ng = [], []
        for ex in tqdm(list(_read(src)), desc=f"selfcheck/{setting}/{g}"):
            ans = (ex.get("greedy_answer") or "").strip()
            samples = [(s.get("answer") or "").strip() for s in ex["samples"]]
            samples = [s for s in samples if s]
            if not ans or not samples:
                continue
            # SelfCheck-NLI: P(contradiction) averaged over samples (higher=hallucination)
            nli_score = float(nli.predict(sentences=[ans], sampled_passages=samples)[0])
            # SelfCheck-1gram: avg neg-logprob of the answer under {answer+samples} unigram LM
            ng = ngram.predict(sentences=[ans], passage=ans, sampled_passages=samples)
            ng_score = float(ng["doc_level"]["avg_neg_logprob"])
            if not np.isfinite(ng_score):
                # unigram LM assigned zero mass (short training-format answers can
                # be entirely novel tokens): maximally surprising -> clamp to the
                # most-hallucinated end instead of crashing the AUROC summary
                ng_score = 1e6
            base = {"example_id": str(ex["example_id"]), "dataset": ex.get("dataset", ""),
                    "split": g, "label": int(ex["label"])}
            recs_nli.append({**base, "helpful_probability": -nli_score, "score": -nli_score})
            recs_ng.append({**base, "helpful_probability": -ng_score, "score": -ng_score})
        _write(recs_nli, out_dir / f"faithful_{setting}__selfcheck_nli__{g}.jsonl")
        _write(recs_ng, out_dir / f"faithful_{setting}__selfcheck_ngram__{g}.jsonl")
        from sklearn.metrics import roc_auc_score
        for name, recs in (("selfcheck_nli", recs_nli), ("selfcheck_ngram", recs_ng)):
            lab = np.array([r["label"] for r in recs]); sc = np.array([r["score"] for r in recs])
            au = float(roc_auc_score(lab, sc)) if lab.min() != lab.max() else float("nan")
            summary.append((name, g, len(recs), float(lab.mean()), au))

    print(f"\n# SelfCheckGPT — {setting}")
    print(f"{'method':18s} {'group':12s} {'N':>5s} {'pos%':>6s} {'AUROC':>7s}")
    for n, g, N, pr, au in sorted(summary, key=lambda x: (x[1], -x[4] if x[4] == x[4] else 0)):
        print(f"{n:18s} {g:12s} {N:5d} {pr*100:5.1f}% {au:7.4f}")


if __name__ == "__main__":
    main()
