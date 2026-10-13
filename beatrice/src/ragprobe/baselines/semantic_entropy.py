"""Semantic Entropy (Kuhn 2023 / Farquhar Nature 2024) and Semantic Entropy
Probes (Kossen 2024) on top of the sampling pass produced by
``scripts/sample_generations.py``.

Pipeline:
  1. Cluster a question's K answers into meaning-classes via bidirectional NLI
     entailment (the ``microsoft/deberta-large-mnli`` model the papers use), with
     a normalized-exact-match fast path for identical short factoid answers.
  2. Semantic entropy:
       - discrete SE: Shannon entropy of the cluster-size distribution (Farquhar's
         "discrete" variant — no log-probs needed, robust for short answers).
       - proper SE: entropy of the cluster-probability distribution where each
         sample is weighted by its length-normalized sequence probability.
  3. SE is a *hallucination* score (high SE -> likely wrong). For correctness
     detection (positive = correct) the score is oriented as ``-SE``.

SEP (``train_sep_probe``): a logistic-regression probe on a single stored hidden
state predicting whether SE is high (above a threshold). At eval time it needs no
sampling — only the hidden state — so it is the cheap, deployable cousin of SE.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

_NLI = {"model": None, "tokenizer": None, "device": None, "entail_idx": None}


# --------------------------------------------------------------------------- #
# Answer normalization / fast-path equality
# --------------------------------------------------------------------------- #
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(s: str) -> str:
    s = (s or "").lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


# --------------------------------------------------------------------------- #
# NLI model (lazy load)
# --------------------------------------------------------------------------- #
def load_nli(model_name: str = "microsoft/deberta-large-mnli", device: str | None = None):
    if _NLI["model"] is not None:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()
    # Find the 'entailment' label index from the model config.
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    entail_idx = next((i for i, v in id2label.items() if "entail" in v), 2)
    _NLI.update(model=model, tokenizer=tok, device=device, entail_idx=entail_idx)


def _entails(premise: str, hypothesis: str, question: str = "") -> bool:
    """Does premise entail hypothesis? (context = the question, per the SE paper)."""
    import torch

    model, tok, device = _NLI["model"], _NLI["tokenizer"], _NLI["device"]
    a = f"{question} {premise}".strip()
    b = f"{question} {hypothesis}".strip()
    inputs = tok(a, b, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    pred = int(torch.argmax(logits).item())
    return pred == _NLI["entail_idx"]


def _bidirectional_equiv(x: str, y: str, question: str) -> bool:
    return _entails(x, y, question) and _entails(y, x, question)


def cluster_answers(answers: list[str], question: str = "", use_nli: bool = True) -> list[int]:
    """Greedy meaning-clustering. Returns a cluster id per answer."""
    cluster_ids = [-1] * len(answers)
    reps: list[int] = []  # index of a representative answer per cluster
    norm = [normalize_answer(a) for a in answers]
    for i, a in enumerate(answers):
        placed = False
        for c, r in enumerate(reps):
            # Fast path: identical normalized strings are always equivalent.
            if norm[i] and norm[i] == norm[r]:
                cluster_ids[i] = c
                placed = True
                break
            if use_nli and _bidirectional_equiv(a, answers[r], question):
                cluster_ids[i] = c
                placed = True
                break
        if not placed:
            cluster_ids[i] = len(reps)
            reps.append(i)
    return cluster_ids


# --------------------------------------------------------------------------- #
# Entropy estimators
# --------------------------------------------------------------------------- #
def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def discrete_semantic_entropy(cluster_ids: list[int]) -> float:
    if not cluster_ids:
        return 0.0
    counts = np.bincount(cluster_ids).astype(float)
    return _entropy(counts / counts.sum())


def proper_semantic_entropy(cluster_ids: list[int], seq_logprobs, num_tokens,
                            length_normalize: bool = True) -> float:
    """SE weighted by per-sample length-normalized sequence probability."""
    if not cluster_ids:
        return 0.0
    lp = np.asarray(seq_logprobs, dtype=float)
    if length_normalize:
        nt = np.asarray(num_tokens, dtype=float)
        nt[nt < 1] = 1.0
        lp = lp / nt
    lp = lp - lp.max()  # stabilize before exp
    w = np.exp(lp)
    n_clusters = max(cluster_ids) + 1
    cluster_mass = np.zeros(n_clusters)
    for cid, wi in zip(cluster_ids, w):
        cluster_mass[cid] += wi
    total = cluster_mass.sum()
    if total <= 0:
        return discrete_semantic_entropy(cluster_ids)
    return _entropy(cluster_mass / total)


# --------------------------------------------------------------------------- #
# Per-example SE record
# --------------------------------------------------------------------------- #
@dataclass
class SERecord:
    example_id: str
    dataset: str
    split: str
    label: int
    question: str = ""
    discrete_se: float = 0.0
    proper_se: float = 0.0
    num_clusters: int = 0
    k: int = 0
    extra: dict = field(default_factory=dict)


def compute_se_for_samples(rows: list[dict], use_nli: bool = True,
                           nli_model: str = "microsoft/deberta-large-mnli") -> list[SERecord]:
    """rows: parsed JSONL records from scripts/sample_generations.py."""
    if use_nli:
        load_nli(nli_model)
    out: list[SERecord] = []
    for r in rows:
        samples = r.get("samples", [])
        answers = [s.get("answer", "") for s in samples]
        question = r.get("prompt", "")  # the prompt already contains the question
        # Use just the trailing question line as NLI context to keep it short.
        q_line = ""
        for line in reversed(question.splitlines()):
            if line.lower().startswith("question:"):
                q_line = line.split(":", 1)[-1].strip()
                break
        cluster_ids = cluster_answers(answers, q_line, use_nli=use_nli)
        d_se = discrete_semantic_entropy(cluster_ids)
        p_se = proper_semantic_entropy(
            cluster_ids,
            [s.get("seq_logprob", 0.0) for s in samples],
            [s.get("num_tokens", 1) for s in samples],
        )
        out.append(SERecord(
            example_id=r["example_id"], dataset=r.get("dataset", ""),
            split=r.get("split", ""), label=int(r.get("label", 0)),
            question=q_line, discrete_se=d_se, proper_se=p_se,
            num_clusters=(max(cluster_ids) + 1 if cluster_ids else 0),
            k=len(samples),
        ))
    return out


# --------------------------------------------------------------------------- #
# Semantic Entropy Probe (hidden state -> high-SE classifier)
# --------------------------------------------------------------------------- #
def train_sep_probe(train_hidden: np.ndarray, train_se: np.ndarray,
                    se_threshold: float | None = None):
    """Logistic regression: hidden -> P(high SE). Returns (model, scaler, threshold)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if se_threshold is None:
        se_threshold = float(np.median(train_se))
    y = (train_se > se_threshold).astype(int)
    scaler = StandardScaler().fit(train_hidden)
    Xs = scaler.transform(train_hidden)
    # Guard against a degenerate single-class split.
    if len(np.unique(y)) < 2:
        return None, scaler, se_threshold
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, y)
    return clf, scaler, se_threshold


def sep_predict_high_se_prob(clf, scaler, hidden: np.ndarray) -> np.ndarray:
    if clf is None:
        return np.zeros(len(hidden))
    return clf.predict_proba(scaler.transform(hidden))[:, 1]
