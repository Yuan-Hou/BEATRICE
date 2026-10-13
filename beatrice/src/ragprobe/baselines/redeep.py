"""ReDeEP scoring, faithful to the official `token_level_reg.py`.

Inputs are the per-example PKS [N,32] (per-layer Knowledge-FFN JSD) and ECS [N,128]
(per copy-head external cosine) produced by scripts/extract_redeep.py. ReDeEP:

  1. Score each PKS layer and ECS head by how well it separates hallucination on a
     calibration split (validation): layers by AUROC(hallucination, pks_layer),
     heads by AUROC(correct, ecs_head)  -- i.e. higher PKS => hallucination, higher
     ECS => faithful, exactly as in the official `calculate_auc_pcc`.
  2. Select the top-K layers and top-N heads, sum each, MinMax-normalize the two sums
     (scaler fit on validation, applied to every split).
  3. ReDeEP hallucination score = m * pks_sum_norm - alpha * ecs_sum_norm   (m=1).

(top_n, top_k, alpha) are chosen on validation by maximizing the final ReDeEP AUROC,
mirroring how the paper fixes per-dataset (i, j, k). We also expose a fully-faithful
PKS-only variant (the Knowledge-FFN half, unaffected by Qwen's 8/32 attention limit).

All returned scores are oriented so HIGHER = more likely CORRECT, to match the other
baselines and the shared evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class RedeepData:
    pks: np.ndarray      # [N, n_layers]
    ecs: np.ndarray      # [N, n_heads]
    label: np.ndarray    # [N]  (1 = correct)
    example_id: np.ndarray
    dataset: np.ndarray
    split: np.ndarray


def _auroc(labels, scores) -> float:
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _minmax_fit(x: np.ndarray):
    lo, hi = float(np.min(x)), float(np.max(x))
    return lo, hi


def _minmax_apply(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (np.clip(x, lo, hi) - lo) / (hi - lo)


def _select_features(val: RedeepData):
    """Rank PKS layers and ECS heads on validation, faithful to calculate_auc_pcc."""
    hall = 1 - val.label  # hallucination label (1 = wrong)
    n_layers, n_heads = val.pks.shape[1], val.ecs.shape[1]
    layer_auc = []  # higher PKS => hallucination
    for k in range(n_layers):
        layer_auc.append((_auroc(hall, val.pks[:, k]), k))
    head_auc = []   # higher ECS => correct (not hallucination)
    for k in range(n_heads):
        head_auc.append((_auroc(val.label, val.ecs[:, k]), k))
    # nan-safe sort descending
    layer_auc.sort(key=lambda t: (-(t[0] if t[0] == t[0] else -1)))
    head_auc.sort(key=lambda t: (-(t[0] if t[0] == t[0] else -1)))
    return [k for _, k in layer_auc], [k for _, k in head_auc], layer_auc, head_auc


def _redeep_hall(data: RedeepData, layers, heads, alpha, lo_p, hi_p, lo_e, hi_e, m=1.0):
    pks_sum = data.pks[:, layers].sum(axis=1)
    ecs_sum = data.ecs[:, heads].sum(axis=1)
    pks_n = _minmax_apply(pks_sum, lo_p, hi_p)
    ecs_n = _minmax_apply(ecs_sum, lo_e, hi_e)
    return m * pks_n - alpha * ecs_n  # higher = hallucination


def run_redeep(val: RedeepData, evalsets: dict[str, RedeepData],
               top_ks=(1, 3, 5, 10, 17, 32), top_ns=(1, 3, 5, 10, 20, 30),
               alphas=(0.1, 0.2, 0.4, 0.6, 1.0)):
    """Returns (scores_by_method, meta). Scores oriented higher=correct."""
    layer_rank, head_rank, layer_auc, head_auc = _select_features(val)
    hall_val = 1 - val.label

    # ---- full ReDeEP: grid-select (top_k layers, top_n heads, alpha) on validation ----
    best = None  # (val_auroc, k, n, alpha, sel_layers, sel_heads, scalers)
    for k in top_ks:
        sel_layers = layer_rank[:k]
        pks_sum_val = val.pks[:, sel_layers].sum(axis=1)
        lo_p, hi_p = _minmax_fit(pks_sum_val)
        for n in top_ns:
            sel_heads = head_rank[:n]
            ecs_sum_val = val.ecs[:, sel_heads].sum(axis=1)
            lo_e, hi_e = _minmax_fit(ecs_sum_val)
            for a in alphas:
                hall = _redeep_hall(val, sel_layers, sel_heads, a, lo_p, hi_p, lo_e, hi_e)
                au = _auroc(hall_val, hall)  # AUROC of hallucination direction
                if au == au and (best is None or au > best[0]):
                    best = (au, k, n, a, sel_layers, sel_heads, (lo_p, hi_p, lo_e, hi_e))
    _, bk, bn, ba, bl, bh, (lo_p, hi_p, lo_e, hi_e) = best

    # ---- PKS-only (fully faithful): select top_k layers maximizing validation AUROC ----
    best_p = None
    for k in top_ks:
        sel = layer_rank[:k]
        s = val.pks[:, sel].sum(axis=1)
        au = _auroc(hall_val, s)
        if au == au and (best_p is None or au > best_p[0]):
            best_p = (au, k, sel)
    _, bpk, bpl = best_p

    scores, meta = {}, {}
    for g, ds in evalsets.items():
        hall = _redeep_hall(ds, bl, bh, ba, lo_p, hi_p, lo_e, hi_e)
        scores.setdefault("redeep", {})[g] = -hall            # orient: higher = correct
        pks_only = ds.pks[:, bpl].sum(axis=1)
        scores.setdefault("redeep_pks", {})[g] = -pks_only    # higher PKS=hallucination
        scores.setdefault("redeep_ecs", {})[g] = ds.ecs[:, bh].sum(axis=1)  # higher=correct
    meta["redeep"] = f"top_k_layers={bk} top_n_heads={bn} alpha={ba} (val_auroc={best[0]:.4f})"
    meta["redeep_pks"] = f"top_k_layers={bpk} layers={sorted(bpl)} (val_auroc={best_p[0]:.4f})"
    meta["redeep_ecs"] = f"top_n_heads={bn} (heads selected with redeep)"
    meta["_layer_auc"] = layer_auc
    meta["_head_auc"] = head_auc[:10]
    return scores, meta
