"""The Tier-A baseline scorers.

Each scorer maps a list of ``ExampleSignals`` to a per-example correctness score where
**higher == more likely correct** (matching the ``helpful_probability`` convention so the
existing evaluator/report can consume the outputs unchanged).

Three families (see docs/baselines.md):
  1. logprob / entropy  — training-free uncertainty scalars (oriented on train).
  2. SAPLMA             — small MLP on the answer-span hidden state (trained).
  3. LLM-Check          — training-free internal-state statistics, plus a trained
                          logistic-regression combiner over those statistics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ragprobe.baselines.extract import ExampleSignals
from ragprobe.baselines.evaluate import binary_detection_metrics
from ragprobe.utils.logging import log


# ---------------------------------------------------------------------------
# 1. Training-free uncertainty scalars (logprob / entropy family + LLM-Check stats)
# ---------------------------------------------------------------------------

def _scalar(sig: ExampleSignals, key: str) -> float:
    p = sig.prob_pool
    if key == "mean_logp":            # higher confidence -> correct
        return -float(p.get("surprisal_mean", 0.0))
    if key == "neg_perplexity":
        return -float(np.exp(min(p.get("surprisal_mean", 0.0), 20.0)))
    if key == "mean_top1":
        return float(p.get("top1_prob_mean", 0.0))
    if key == "min_top1":             # worst (most surprising) token
        return float(p.get("top1_prob_min", 0.0))
    if key == "neg_mean_entropy":
        return -float(p.get("entropy_mean", 0.0))
    if key == "max_entropy":          # peak uncertainty over the answer
        return -float(p.get("entropy_max", 0.0))
    if key == "mean_margin":
        return float(p.get("margin_mean", 0.0))
    if key == "llmcheck_hidden_logdet":
        return float(sig.hidden_logdet)
    raise KeyError(key)


TRAINING_FREE_SCALARS = (
    "mean_logp",
    "neg_perplexity",
    "mean_top1",
    "min_top1",
    "neg_mean_entropy",
    "max_entropy",
    "mean_margin",
    "llmcheck_hidden_logdet",
)


def score_training_free(key: str, train: list[ExampleSignals], evalsets: dict[str, list[ExampleSignals]]) -> dict[str, np.ndarray]:
    """Compute a scalar score and fix its orientation using the train split (1 bit)."""
    train_scores = np.array([_scalar(s, key) for s in train], dtype=np.float64)
    train_labels = np.array([s.label for s in train], dtype=np.int64)
    sign = 1.0
    if len(set(train_labels.tolist())) == 2:
        auroc = binary_detection_metrics(train_labels, train_scores).get("auroc")
        if auroc is not None and auroc < 0.5:
            sign = -1.0
    return {name: sign * np.array([_scalar(s, key) for s in rows], dtype=np.float64)
            for name, rows in evalsets.items()}


# ---------------------------------------------------------------------------
# 2. SAPLMA: MLP on the answer-span hidden state
# ---------------------------------------------------------------------------

class _SAPLMANet(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _hidden_matrix(rows: list[ExampleSignals]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    feats, labels, keep = [], [], []
    for i, s in enumerate(rows):
        if s.answer_hidden_mean is None:
            continue
        feats.append(s.answer_hidden_mean)
        labels.append(s.label)
        keep.append(i)
    if not feats:
        return np.zeros((0, 0), np.float32), np.zeros((0,), np.int64), keep
    return np.stack(feats).astype(np.float32), np.array(labels, np.int64), keep


def score_saplma(
    train: list[ExampleSignals],
    evalsets: dict[str, list[ExampleSignals]],
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    device: str | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, _ = _hidden_matrix(train)
    if Xtr.shape[0] == 0:
        raise RuntimeError("SAPLMA: no answer hidden states in train split")
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)

    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    model = _SAPLMANet(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.from_numpy(Xtr_s)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).unsqueeze(1)
    n = Xtr_t.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    out: dict[str, np.ndarray] = {}
    for name, rows in evalsets.items():
        X, _, keep = _hidden_matrix(rows)
        scores = np.full(len(rows), np.nan, dtype=np.float64)
        if X.shape[0] > 0:
            Xs = scaler.transform(X).astype(np.float32)
            with torch.no_grad():
                logits = model(torch.from_numpy(Xs).to(device)).squeeze(1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            for j, i in enumerate(keep):
                scores[i] = probs[j]
        # examples without hidden states get the prior (mean train label)
        scores = np.where(np.isnan(scores), float(ytr.mean()), scores)
        out[name] = scores
    return out


# ---------------------------------------------------------------------------
# 3. LLM-Check supervised combiner: logistic regression over internal-state stats
# ---------------------------------------------------------------------------

LLMCHECK_FEATURES = (
    "surprisal_mean", "surprisal_max", "entropy_mean", "entropy_max",
    "top1_prob_mean", "top1_prob_min", "margin_mean", "p_mean",
)


def _llmcheck_matrix(rows: list[ExampleSignals]) -> np.ndarray:
    feats = []
    for s in rows:
        vec = [float(s.prob_pool.get(k, 0.0)) for k in LLMCHECK_FEATURES]
        vec.append(float(s.hidden_logdet))
        feats.append(vec)
    return np.asarray(feats, dtype=np.float64)


def score_logreg_combiner(
    train: list[ExampleSignals],
    evalsets: dict[str, list[ExampleSignals]],
    feature_fn=_llmcheck_matrix,
) -> dict[str, np.ndarray]:
    Xtr = feature_fn(train)
    ytr = np.array([s.label for s in train], dtype=np.int64)
    if len(set(ytr.tolist())) < 2:
        raise RuntimeError("logreg combiner: train split has a single class")
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    return {name: clf.predict_proba(scaler.transform(feature_fn(rows)))[:, 1]
            for name, rows in evalsets.items()}


def _prob_pool_matrix(rows: list[ExampleSignals]) -> np.ndarray:
    keys = sorted({k for s in rows for k in s.prob_pool})
    return np.asarray([[float(s.prob_pool.get(k, 0.0)) for k in keys] for s in rows], dtype=np.float64)


# Registry: name -> (kind, callable). kind in {"free", "trained"}.
def build_scorers() -> dict:
    scorers: dict[str, tuple[str, object]] = {}
    for key in TRAINING_FREE_SCALARS:
        scorers[key] = ("free", key)
    scorers["saplma_mlp"] = ("trained", "saplma")
    scorers["logreg_prob"] = ("trained", "logreg_prob")
    scorers["llmcheck_combo"] = ("trained", "llmcheck_combo")
    return scorers


def run_scorer(name: str, kind: str, payload, train, evalsets) -> dict[str, np.ndarray]:
    if kind == "free":
        return score_training_free(payload, train, evalsets)
    if payload == "saplma":
        return score_saplma(train, evalsets)
    if payload == "logreg_prob":
        return score_logreg_combiner(train, evalsets, feature_fn=_prob_pool_matrix)
    if payload == "llmcheck_combo":
        return score_logreg_combiner(train, evalsets, feature_fn=_llmcheck_matrix)
    raise KeyError(f"unknown scorer {name}")
