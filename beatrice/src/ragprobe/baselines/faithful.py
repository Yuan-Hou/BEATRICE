"""Paper-faithful baselines built on the re-extracted per-layer features
(``data/layer_features/<setting>/<group>__shard*.npz`` from
``scripts/extract_layer_features.py``):

  - SAPLMA (Azaria & Mitchell 2023): MLP[256,128,64] on a SINGLE token's hidden,
    with a LAYER SWEEP; best layer selected by validation AUROC. Last-answer-token
    is the default position; mean-pool is offered as an ablation.
  - SEP (Kossen 2024): logistic-regression probe on a single token's hidden,
    predicting binarized Semantic Entropy; layer swept, selected on validation.
  - LLM-Check (Sriramanan 2024): per-layer hidden log-det (eigen) score +
    per-attention-layer attention log-det score + perplexity + entropy, each as an
    individual training-free detector and as a logistic combination.

All selection happens on the `validation` group; `test_id` and `popqa_ood` are
reported untouched. Returns/writes scores oriented so higher = more likely correct.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class LayerData:
    example_id: np.ndarray          # [N] str
    dataset: np.ndarray             # [N] str
    label: np.ndarray               # [N] int
    hidden_last: np.ndarray         # [N, L+1, d] fp16
    hidden_mean: np.ndarray         # [N, L+1, d] fp16
    hid_logdet: np.ndarray          # [N, L+1]
    attn_logdet: np.ndarray | None  # [N, A]
    perplexity: np.ndarray          # [N]
    mean_entropy: np.ndarray        # [N]
    max_entropy: np.ndarray         # [N]


def load_layer_features(setting: str, group: str, base: str = "data/layer_features") -> LayerData:
    d = os.path.join(base, setting)
    npz_paths = sorted(glob.glob(os.path.join(d, f"{group}__shard*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"no shards for {setting}/{group} under {d}")
    parts, index = [], []
    for p in npz_paths:
        parts.append(np.load(p))
        idx_path = p.replace(".npz", ".index.jsonl")
        index.extend(json.loads(l) for l in open(idx_path))

    def cat(key):
        arrs = [pt[key] for pt in parts if key in pt.files and pt[key].size > 0]
        return np.concatenate(arrs, axis=0) if arrs else None

    has_attn = all("attn_logdet" in pt.files for pt in parts)
    return LayerData(
        example_id=np.array([r["example_id"] for r in index]),
        dataset=np.array([r["dataset"] for r in index]),
        label=np.array([int(r["label"]) for r in index]),
        hidden_last=cat("hidden_last").astype(np.float32),
        hidden_mean=cat("hidden_mean").astype(np.float32),
        hid_logdet=cat("hid_logdet"),
        attn_logdet=(cat("attn_logdet") if has_attn else None),
        perplexity=cat("perplexity"), mean_entropy=cat("mean_entropy"),
        max_entropy=cat("max_entropy"),
    )


def concat_layerdata(a: LayerData, b: LayerData) -> LayerData:
    """Concatenate two splits (for training augmentation). Only the fields SAPLMA
    needs are guaranteed meaningful; attn_logdet kept if both have it."""
    attn = None
    if a.attn_logdet is not None and b.attn_logdet is not None:
        attn = np.concatenate([a.attn_logdet, b.attn_logdet], 0)
    return LayerData(
        example_id=np.concatenate([a.example_id, b.example_id]),
        dataset=np.concatenate([a.dataset, b.dataset]),
        label=np.concatenate([a.label, b.label]),
        hidden_last=np.concatenate([a.hidden_last, b.hidden_last], 0),
        hidden_mean=np.concatenate([a.hidden_mean, b.hidden_mean], 0),
        hid_logdet=np.concatenate([a.hid_logdet, b.hid_logdet], 0),
        attn_logdet=attn,
        perplexity=np.concatenate([a.perplexity, b.perplexity]),
        mean_entropy=np.concatenate([a.mean_entropy, b.mean_entropy]),
        max_entropy=np.concatenate([a.max_entropy, b.max_entropy]),
    )


def load_se_cache(setting: str, group: str, base: str = "data/se_samples") -> dict:
    # SE_CACHE_BASE lets the aligned re-run point SE/SEP at the training-format
    # resamples (data/se_samples_trainfmt) without changing call sites.
    base = os.environ.get("SE_CACHE_BASE", base)
    path = os.path.join(base, setting, f"{group}.se.jsonl")
    out = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out[r["example_id"]] = (r["discrete_se"], r["proper_se"])
    return out


# --------------------------------------------------------------------------- #
# Metrics helper
# --------------------------------------------------------------------------- #
def auroc(labels, scores) -> float:
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# --------------------------------------------------------------------------- #
# SAPLMA: MLP[256,128,64] on one token's hidden, layer-swept
# --------------------------------------------------------------------------- #
def _train_mlp(Xtr, ytr, Xva, device="cpu", epochs=40, lr=1e-3, wd=1e-4, bs=512, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pw = torch.tensor([neg / max(pos, 1.0)], device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = Xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xtr[idx]).squeeze(1)
            loss = lossf(out, ytr_t[idx])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Xva_t = torch.tensor(sc.transform(Xva), dtype=torch.float32, device=device)
        p = torch.sigmoid(net(Xva_t).squeeze(1)).cpu().numpy()
    return net, sc, p


def _mlp_predict(net, sc, X, device="cpu"):
    import torch
    net.eval()
    with torch.no_grad():
        Xt = torch.tensor(sc.transform(X), dtype=torch.float32, device=device)
        return torch.sigmoid(net(Xt).squeeze(1)).cpu().numpy()


def run_saplma(train: LayerData, evalsets: dict, pooling="last", device="cpu"):
    """Layer sweep selected on validation AUROC. Returns (scores_by_group, best_layer)."""
    Xtr_all = train.hidden_last if pooling == "last" else train.hidden_mean
    val = evalsets["validation"]
    Xva_all = val.hidden_last if pooling == "last" else val.hidden_mean
    n_layers = Xtr_all.shape[1]
    best_layer, best_auc, best_net, best_sc = -1, -1.0, None, None
    per_layer = []
    for l in range(n_layers):
        net, sc, pva = _train_mlp(Xtr_all[:, l], train.label.astype(np.float32),
                                  Xva_all[:, l], device=device)
        a = auroc(val.label, pva)
        per_layer.append(a)
        if a > best_auc:
            best_auc, best_layer, best_net, best_sc = a, l, net, sc
    scores = {}
    for name, ds in evalsets.items():
        X = (ds.hidden_last if pooling == "last" else ds.hidden_mean)[:, best_layer]
        scores[name] = _mlp_predict(best_net, best_sc, X, device=device)
    return scores, best_layer, per_layer


# --------------------------------------------------------------------------- #
# LLMsKnow (Orgad 2024): LogisticRegression on the EXACT-ANSWER token's hidden,
# layer-swept. Their probe_specific_layer_token uses LogisticRegression on a single
# (layer, token); the headline token is the exact-answer last token, which for our
# short-answer QA is exactly `hidden_last` (last token of the answer span).
# --------------------------------------------------------------------------- #
def run_llmsknow(train: LayerData, evalsets: dict, device="cpu"):
    """LogisticRegression on exact-answer-token hidden, best layer by validation AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr_all = train.hidden_last  # exact-answer last token
    val = evalsets["validation"]
    Xva_all = val.hidden_last
    n_layers = Xtr_all.shape[1]
    best_layer, best_auc, best_clf, best_sc = -1, -1.0, None, None
    per_layer = []
    for l in range(n_layers):
        sc = StandardScaler().fit(Xtr_all[:, l])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr_all[:, l]), train.label)
        pva = clf.predict_proba(sc.transform(Xva_all[:, l]))[:, 1]
        a = auroc(val.label, pva)
        per_layer.append(a)
        if a > best_auc:
            best_auc, best_layer, best_clf, best_sc = a, l, clf, sc
    scores = {}
    for name, ds in evalsets.items():
        X = best_sc.transform(ds.hidden_last[:, best_layer])
        scores[name] = best_clf.predict_proba(X)[:, 1]
    return scores, best_layer, per_layer


# --------------------------------------------------------------------------- #
# SEP: logistic probe on one token's hidden -> binarized SE, layer-swept
# --------------------------------------------------------------------------- #
def run_sep(train_sources, evalsets: dict, pooling="last", se_key="discrete", device="cpu"):
    """train_sources: list of (LayerData, se_dict). Each source aligns rows to its OWN
    SE cache by id (original and rewritten share ids, so they must stay separate)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    se_idx = 0 if se_key == "discrete" else 1
    Xparts, separts = [], []
    for ld, se_dict in train_sources:
        Xall = ld.hidden_last if pooling == "last" else ld.hidden_mean
        se = np.array([se_dict.get(eid, (np.nan, np.nan))[se_idx] for eid in ld.example_id])
        keep = ~np.isnan(se)
        Xparts.append(Xall[keep])
        separts.append(se[keep])
    Xtr_all = np.concatenate(Xparts, 0)
    se_vals = np.concatenate(separts)
    thr = float(np.median(se_vals))
    y_high = (se_vals > thr).astype(int)
    val = evalsets["validation"]
    Xva_all = val.hidden_last if pooling == "last" else val.hidden_mean
    n_layers = Xtr_all.shape[1]
    if len(np.unique(y_high)) < 2:
        return {k: np.full(len(v.label), 0.5) for k, v in evalsets.items()}, -1, []
    best_layer, best_auc, best_clf, best_sc = -1, -1.0, None, None
    per_layer = []
    for l in range(n_layers):
        sc = StandardScaler().fit(Xtr_all[:, l])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr_all[:, l]), y_high)
        # correctness score = 1 - P(high SE); select layer by val correctness-AUROC
        pva = 1.0 - clf.predict_proba(sc.transform(Xva_all[:, l]))[:, 1]
        a = auroc(val.label, pva)
        per_layer.append(a)
        if a > best_auc:
            best_auc, best_layer, best_clf, best_sc = a, l, clf, sc
    scores = {}
    for name, ds in evalsets.items():
        X = (ds.hidden_last if pooling == "last" else ds.hidden_mean)[:, best_layer]
        scores[name] = 1.0 - best_clf.predict_proba(best_sc.transform(X))[:, 1]
    return scores, best_layer, per_layer


# --------------------------------------------------------------------------- #
# LLM-Check: hidden / attention / perplexity / entropy + logistic combination
# --------------------------------------------------------------------------- #
def _san(x):
    """Make finite while preserving rank: +inf -> max+1, -inf -> min-1, nan -> median.
    (slogdet can be -inf on singular Gram; exp(surprisal) can overflow to +inf.)"""
    x = np.asarray(x, dtype=np.float64).copy()
    fin = x[np.isfinite(x)]
    if fin.size == 0:
        return np.zeros_like(x)
    x[np.isposinf(x)] = fin.max() + 1.0
    x[np.isneginf(x)] = fin.min() - 1.0
    x[np.isnan(x)] = np.median(fin)
    return x


def _orient(train_scores, train_labels, eval_scores):
    """Flip sign so higher = more likely correct, decided on train AUROC."""
    a = auroc(train_labels, _san(train_scores))
    flip = (a < 0.5)
    return {k: _san(-v if flip else v) for k, v in eval_scores.items()}


def run_llmcheck(train: LayerData, evalsets: dict):
    out = {}

    def mean_layer(ld, attr, cols=None):
        m = getattr(ld, attr)
        if cols is not None:
            m = m[:, cols]
        return np.nanmean(m, axis=1)

    # Hidden score (mean over all layers of the log-det)
    tr = mean_layer(train, "hid_logdet")
    ev = {k: mean_layer(v, "hid_logdet") for k, v in evalsets.items()}
    out["llmcheck_hidden"] = _orient(tr, train.label, ev)

    # Attention score (mean over the full-attention layers)
    if train.attn_logdet is not None and all(v.attn_logdet is not None for v in evalsets.values()):
        tr = np.nanmean(train.attn_logdet, axis=1)
        ev = {k: np.nanmean(v.attn_logdet, axis=1) for k, v in evalsets.items()}
        out["llmcheck_attention"] = _orient(tr, train.label, ev)

    # Perplexity (lower ppl -> more confident)
    out["llmcheck_perplexity"] = _orient(train.perplexity, train.label,
                                         {k: v.perplexity for k, v in evalsets.items()})
    # Entropy
    out["llmcheck_entropy"] = _orient(train.mean_entropy, train.label,
                                      {k: v.mean_entropy for k, v in evalsets.items()})

    # Logistic combination of all LLM-Check features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def feats(ld):
        cols = [ld.hid_logdet, ld.perplexity[:, None], ld.mean_entropy[:, None], ld.max_entropy[:, None]]
        if ld.attn_logdet is not None:
            cols.insert(1, ld.attn_logdet)
        m = np.concatenate(cols, axis=1)
        return np.apply_along_axis(_san, 0, m)  # sanitize each feature column

    Xtr = feats(train)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), train.label)
    out["llmcheck_combined"] = {k: clf.predict_proba(sc.transform(feats(v)))[:, 1]
                                for k, v in evalsets.items()}
    return out


# --------------------------------------------------------------------------- #
# SE direct baseline (from the cache), aligned to the layer-feature example order
# --------------------------------------------------------------------------- #
def se_scores(evalsets: dict, setting: str, se_key="discrete"):
    idx = 0 if se_key == "discrete" else 1
    out = {}
    for name, ds in evalsets.items():
        cache = load_se_cache(setting, name)
        se = _san(np.array([cache.get(eid, (np.nan, np.nan))[idx] for eid in ds.example_id]))
        mu, sdv = np.nanmean(se), np.nanstd(se) + 1e-8
        out[name] = 1.0 / (1.0 + np.exp((se - mu) / sdv))   # high SE -> low P(correct)
    return out
