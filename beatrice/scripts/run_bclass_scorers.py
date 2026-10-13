"""Turn EigenScore and Lookback-Lens features into standard-schema predictions.

  EigenScore : data/eigenscore/<setting>/<group>.jsonl  ({eigenscore})  -- training-free,
               helpful = -eigenscore (higher dispersion => hallucination).
  Lookback   : data/lookback_features/withctx/<group>.npz ({feat[N,128]}) -- a logistic
               regression detector (faithful to step03) fit on validation; test_id /
               popqa_ood are held out; the validation row is 5-fold CV (no in-sample
               optimism). withctx only.

    python scripts/run_bclass_scorers.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ragprobe.baselines import aligned_labels as AL

GROUPS = ["validation", "test_id", "popqa_ood"]


def _auroc(lab, sc):
    lab = np.asarray(lab)
    return float(roc_auc_score(lab, sc)) if lab.min() != lab.max() else float("nan")


def _write(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def score_eigenscore(eig_base="data/eigenscore", out_root=None):
    summary = []
    for setting in ("noctx", "withctx"):
        out_dir = Path(f"{out_root or AL.out_root()}/baselines/faithful/{setting}")
        for g in GROUPS:
            src = Path(f"{eig_base}/{setting}/{g}.jsonl")
            if not src.exists():
                continue
            rows = _read_jsonl(src)
            recs = [{"example_id": str(r["example_id"]), "dataset": r.get("dataset", ""),
                     "split": g, "label": int(r["label"]),
                     "helpful_probability": -float(r["eigenscore"]), "score": -float(r["eigenscore"])}
                    for r in rows]
            recs = AL.override(recs, setting, g)
            _write(recs, out_dir / f"faithful_{setting}__eigenscore__{g}.jsonl")
            lab = [r["label"] for r in recs]; sc = [r["score"] for r in recs]
            summary.append(("eigenscore", setting, g, len(recs), _auroc(lab, sc)))
    return summary


def _load_lookback_group(base: Path, g: str):
    """Load a lookback group's features + index. Sharded (train/train_rewritten) or single file."""
    shards = sorted(glob.glob(str(base / f"{g}__shard*.npz")))
    if shards:
        feats, idx = [], []
        for sh in shards:
            feats.append(np.load(sh)["feat"])
            idx.extend(_read_jsonl(sh.replace(".npz", ".index.jsonl")))
        return (np.concatenate(feats, 0) if feats else None), idx
    p = base / f"{g}.npz"
    if not p.exists():
        return None, None
    return np.load(p)["feat"], _read_jsonl(base / f"{g}.index.jsonl")


def score_lookback(lb_base="data/lookback_features/withctx", out_root=None):
    """Faithful Lookback-Lens: fit the logistic detector on the SAME training data the other
    trained baselines use — train (+ rewritten-context aug, 30k+30k) — and evaluate on
    validation/test/ood as fully held-out groups. (The earlier version trained on ~1.2k
    validation only, which under-powered this trained baseline; see docs/baseline-fidelity.)"""
    base = Path(lb_base)
    Xtr_parts, ytr_parts = [], []
    for tg in ("train", "train_rewritten"):
        f, idx = _load_lookback_group(base, tg)
        if f is not None and len(f):
            if AL.enabled():
                # optional label override: base rows -> withctx verdicts, rewritten
                # rows -> rewritten-context verdicts; out-of-scope rows drop
                setting = "withctx" if tg == "train" else "rewritten"
                keep, labels = AL.override_mask([r["example_id"] for r in idx], setting, tg)
                keep = np.array(keep)
                f = f[keep]
                ytr_parts.append(np.array(labels)[keep])
            else:
                ytr_parts.append(np.array([r["label"] for r in idx]))
            Xtr_parts.append(f)
    if not Xtr_parts:
        return []  # faithful path requires train features; skip if not yet extracted
    Xtr = np.concatenate(Xtr_parts, 0)
    ytr = np.concatenate(ytr_parts)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(scaler.transform(Xtr), ytr)

    out_dir = Path(f"{out_root or AL.out_root()}/baselines/faithful/withctx")
    summary = []
    for g in GROUPS:
        feat, idx = _load_lookback_group(base, g)
        if feat is None:
            continue
        if AL.enabled():
            keep, labels = AL.override_mask([r["example_id"] for r in idx], "withctx", g)
            keep = np.array(keep)
            feat = feat[keep]
            idx = [r for r, k in zip(idx, keep) if k]
            lab = np.array(labels)[keep]
        else:
            lab = np.array([r["label"] for r in idx])
        sc = clf.predict_proba(scaler.transform(feat))[:, 1]
        recs = [{"example_id": str(idx[i]["example_id"]), "dataset": idx[i].get("dataset", ""),
                 "split": g, "label": int(lab[i]),
                 "helpful_probability": float(sc[i]), "score": float(sc[i])} for i in range(len(lab))]
        _write(recs, out_dir / f"faithful_withctx__lookback_lens__{g}.jsonl")
        summary.append(("lookback_lens", "withctx", g, len(recs), _auroc(lab, sc)))
    return summary


def main():
    global GROUPS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=GROUPS,
                    help="Which eval groups to process (default: %(default)s).")
    ap.add_argument("--eigenscore-base", default="data/eigenscore")
    ap.add_argument("--lookback-base", default="data/lookback_features/withctx")
    ap.add_argument("--out-root", default=None,
                    help="prediction root (gemma: outputs/gemma4/predictions)")
    args = ap.parse_args()
    GROUPS = list(args.groups)

    summary = (score_eigenscore(args.eigenscore_base, args.out_root)
               + score_lookback(args.lookback_base, args.out_root))
    print(f"\n{'method':14s} {'setting':8s} {'group':12s} {'N':>5s} {'AUROC':>7s}")
    for m, s, g, n, au in summary:
        print(f"{m:14s} {s:8s} {g:12s} {n:5d} {au:7.4f}")


if __name__ == "__main__":
    main()
