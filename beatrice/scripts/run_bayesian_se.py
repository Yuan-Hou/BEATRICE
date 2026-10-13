"""Bayesian Semantic Entropy (Aichberger / cf. 2504.03579) baseline.

Standard Semantic Entropy uses the plug-in (MLE) entropy of the NLI-cluster
proportions, which is downward-biased at small sample size K. The Bayesian variant
puts a Dirichlet prior on the cluster probabilities and uses the POSTERIOR-MEAN
entropy (Wolpert & Wolf closed form), which corrects that bias and (per the paper)
reaches the same AUROC at a much smaller sampling budget:

    E[H] = psi(N + C*b + 1) - sum_c (n_c + b)/(N + C*b) * psi(n_c + b + 1)

with counts n_c over the C observed meaning-clusters, N = K, b = 1 (Laplace), psi =
digamma (nats). We reuse the exact same NLI clustering as Semantic Entropy
(microsoft/deberta-large-mnli, bidirectional entailment). Higher = more dispersed =
hallucination; written scores are oriented higher = CORRECT.

NOTE (deviation): no official code was found for 2504.03579; we reproduce the
*Bayesian-estimator-over-SE-clusters* idea with the Wolpert-Wolf Dirichlet posterior
mean. At full K=10 this tracks plain SE closely (the paper's gain is sample budget).

    CUDA_VISIBLE_DEVICES=2 .../vllm2/bin/python scripts/run_bayesian_se.py --setting withctx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import digamma
from tqdm import tqdm

from ragprobe.baselines.semantic_entropy import cluster_answers, load_nli
from ragprobe.utils.logging import log

GROUPS = ["validation", "test_id", "popqa_ood"]


def bayesian_entropy(counts: np.ndarray, beta: float = 1.0) -> float:
    """Wolpert-Wolf posterior-mean entropy under symmetric Dirichlet(beta), nats."""
    counts = counts.astype(float)
    N = counts.sum()
    C = len(counts)
    if N < 1 or C < 2:
        return 0.0
    A = C * beta
    p = (counts + beta) / (N + A)
    return float(digamma(N + A + 1) - np.sum(p * digamma(counts + beta + 1)))


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
    ap.add_argument("--groups", nargs="*", default=GROUPS,
                    help="Which eval groups to process (default: %(default)s).")
    ap.add_argument("--samples-dir", default=None,
                    help="Override the samples dir (e.g. training-format resamples).")
    ap.add_argument("--out-dir", default=None,
                    help="Override output dir (gemma runs must not clobber qwen predictions).")
    args = ap.parse_args()
    GROUPS = list(args.groups)
    setting = args.setting
    from ragprobe.baselines import aligned_labels as AL
    out_dir = Path(args.out_dir or f"{AL.out_root()}/baselines/faithful/{setting}")
    sdir = args.samples_dir or f"data/se_samples/{setting}"
    load_nli(device=args.device)

    from sklearn.metrics import roc_auc_score
    summary = []
    for g in GROUPS:
        rows = [json.loads(l) for l in open(f"{sdir}/{g}.jsonl") if l.strip()]
        recs = []
        for ex in tqdm(rows, desc=f"bayes_se/{setting}/{g}"):
            answers = [(s.get("answer") or "").strip() for s in ex["samples"]]
            answers = [a for a in answers if a]
            if len(answers) < 2:
                continue
            cluster_ids = cluster_answers(answers, question="", use_nli=True)
            counts = np.bincount(cluster_ids).astype(float)
            bse = bayesian_entropy(counts)
            recs.append({"example_id": str(ex["example_id"]), "dataset": ex.get("dataset", ""),
                         "split": g, "label": int(ex["label"]),
                         "helpful_probability": -bse, "score": -bse})
        _write(recs, out_dir / f"faithful_{setting}__bayesian_se__{g}.jsonl")
        lab = np.array([r["label"] for r in recs]); sc = np.array([r["score"] for r in recs])
        au = float(roc_auc_score(lab, sc)) if lab.min() != lab.max() else float("nan")
        summary.append((g, len(recs), float(lab.mean()), au))

    print(f"\n# Bayesian SE — {setting}")
    for g, n, pr, au in summary:
        print(f"{g:12s} N={n:5d} pos={pr*100:4.1f}% AUROC={au:.4f}")


if __name__ == "__main__":
    main()
