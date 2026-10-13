"""Score the Tier-C RAG baselines (ReDeEP + LUMINA) from their extracted features
and write predictions in the standard schema so they fold into the same-sample
comparison and the shared evaluator.

  ReDeEP : data/redeep_features/withctx/<group>__shard*.npz  (pks, ecs, head_ids)
  LUMINA : data/lumina_features/withctx/<group>.npz          (mmd, ipr)

Selection / hyperparameters (ReDeEP top-K/top-N/alpha, LUMINA lambda) are calibrated
on the `validation` group; test_id and popqa_ood are held out. All written scores are
oriented higher = more likely CORRECT.

    python scripts/run_tierc_baselines.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from ragprobe.baselines import aligned_labels as AL
from ragprobe.baselines.redeep import RedeepData, run_redeep
from ragprobe.utils.logging import log

GROUPS = ["validation", "test_id", "popqa_ood"]
REDEEP_DIR = "data/redeep_features/withctx"
LUMINA_DIR = "data/lumina_features/withctx"
OUT_DIR = Path(f"{AL.out_root()}/baselines/faithful/withctx")  # overridable via --dirs


def _auroc(labels, scores) -> float:
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _read_index(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_redeep(group) -> RedeepData:
    shards = sorted(glob.glob(f"{REDEEP_DIR}/{group}__shard*.npz"))
    pks, ecs, idx = [], [], []
    for sh in shards:
        d = np.load(sh)
        pks.append(d["pks"]); ecs.append(d["ecs"])
        idx.extend(_read_index(sh.replace(".npz", ".index.jsonl")))
    pks = np.concatenate(pks, 0); ecs = np.concatenate(ecs, 0)
    labels = np.array([r["label"] for r in idx], dtype=np.int64)
    if AL.enabled():
        keep, new_lab = AL.override_mask([r["example_id"] for r in idx], "withctx", group)
        keep = np.array(keep)
        pks, ecs = pks[keep], ecs[keep]
        labels = np.array(new_lab, dtype=np.int64)[keep]
        idx = [r for r, k in zip(idx, keep) if k]
    return RedeepData(
        pks=pks, ecs=ecs,
        label=labels,
        example_id=np.array([r["example_id"] for r in idx]),
        dataset=np.array([r["dataset"] for r in idx]),
        split=np.array([r["split"] for r in idx]),
    )


def _load_lumina(group):
    d = np.load(f"{LUMINA_DIR}/{group}.npz")
    idx = _read_index(f"{LUMINA_DIR}/{group}.index.jsonl")
    mmd, ipr = d["mmd"], d["ipr"]
    labels = np.array([r["label"] for r in idx], dtype=np.int64)
    if AL.enabled():
        keep, new_lab = AL.override_mask([r["example_id"] for r in idx], "withctx", group)
        keep = np.array(keep)
        mmd, ipr = mmd[keep], ipr[keep]
        labels = np.array(new_lab, dtype=np.int64)[keep]
        idx = [dict(r, label=int(l)) for r, k, l in zip(idx, keep, np.array(new_lab)) if k]
    return (mmd, ipr, labels, [r for r in idx])


def _write(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_lumina_combine(lum):
    """Tune lambda on validation; return scores_by_method (higher=correct) + meta."""
    mmd_v, ipr_v, lab_v, _ = lum["validation"]
    hall_v = 1 - lab_v
    best = None
    for lam in np.linspace(0, 1, 11):
        hall = lam * ipr_v - (1 - lam) * mmd_v
        au = _auroc(hall_v, hall)
        if au == au and (best is None or au > best[0]):
            best = (au, float(lam))
    _, lam = best
    scores = {"lumina": {}, "lumina_mmd": {}, "lumina_ipr": {}}
    for g in GROUPS:
        mmd, ipr, _, _ = lum[g]
        scores["lumina"][g] = -(lam * ipr - (1 - lam) * mmd)  # higher=correct
        scores["lumina_mmd"][g] = mmd                          # higher context use=correct
        scores["lumina_ipr"][g] = -ipr                         # higher internal=hallucination
    return scores, {"lumina": f"lambda={lam:.2f} (val_auroc={best[0]:.4f})"}


def main():
    global GROUPS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=GROUPS,
                    help="Which eval groups to process (default: %(default)s).")
    ap.add_argument("--redeep-dir", default=None)
    ap.add_argument("--lumina-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    GROUPS = list(args.groups)
    global REDEEP_DIR, LUMINA_DIR, OUT_DIR
    if args.redeep_dir:
        REDEEP_DIR = args.redeep_dir
    if args.lumina_dir:
        LUMINA_DIR = args.lumina_dir
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    # ---- ReDeEP ----
    rd = {g: _load_redeep(g) for g in GROUPS}
    rd_scores, rd_meta = run_redeep(rd["validation"], rd)
    log(f"[redeep] {rd_meta['redeep']}")
    log(f"[redeep_pks] {rd_meta['redeep_pks']}")

    # ---- LUMINA ----
    lum = {g: _load_lumina(g) for g in GROUPS}
    lum_scores, lum_meta = run_lumina_combine(lum)
    log(f"[lumina] {lum_meta['lumina']}")

    # index maps for writing
    rd_idx = {g: rd[g] for g in GROUPS}
    lum_idx = {g: lum[g][3] for g in GROUPS}

    summary = []
    # write ReDeEP methods
    for mname, by_group in rd_scores.items():
        for g in GROUPS:
            ds = rd_idx[g]; sc = by_group[g]
            recs = [{
                "example_id": str(ds.example_id[i]), "dataset": str(ds.dataset[i]),
                "split": g, "label": int(ds.label[i]),
                "helpful_probability": float(sc[i]), "score": float(sc[i]),
            } for i in range(len(ds.label))]
            _write(recs, OUT_DIR / f"faithful_withctx__{mname}__{g}.jsonl")
            summary.append((mname, g, len(recs), float(ds.label.mean()), _auroc(ds.label, sc)))
    # write LUMINA methods
    for mname, by_group in lum_scores.items():
        for g in GROUPS:
            idx = lum_idx[g]; sc = by_group[g]
            lab = np.array([r["label"] for r in idx])
            recs = [{
                "example_id": str(idx[i]["example_id"]), "dataset": str(idx[i]["dataset"]),
                "split": g, "label": int(idx[i]["label"]),
                "helpful_probability": float(sc[i]), "score": float(sc[i]),
            } for i in range(len(idx))]
            _write(recs, OUT_DIR / f"faithful_withctx__{mname}__{g}.jsonl")
            summary.append((mname, g, len(recs), float(lab.mean()), _auroc(lab, sc)))

    all_meta = {**rd_meta, **lum_meta}
    lines = ["# Tier-C baselines — withctx (ReDeEP + LUMINA)", ""]
    for k in ("redeep", "redeep_pks", "redeep_ecs", "lumina"):
        if k in all_meta:
            lines.append(f"{k}: {all_meta[k]}")
    lines.append("")
    lines.append(f"{'method':22s} {'group':12s} {'N':>5s} {'pos%':>6s} {'AUROC':>7s}")
    for mname, g, n, pr, au in sorted(summary, key=lambda x: (x[1], -x[4] if x[4] == x[4] else 0)):
        lines.append(f"{mname:22s} {g:12s} {n:5d} {pr*100:5.1f}% {au:7.4f}")
    text = "\n".join(lines)
    (OUT_DIR / "tierc_withctx__summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
