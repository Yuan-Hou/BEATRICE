"""Per-example signal extraction for the correctness-detection baselines.

We deliberately reuse the same loaders/labels as ``ragprobe.training.dataset`` so a
baseline scores the *identical* examples (and identical binary labels) as the main
probes. Nothing here loads the LLM: every signal comes from features that were
already dumped to disk by ``scripts/extract_features.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from ragprobe.training.dataset import (
    _get_example,
    _get_feature_names,
    _get_sidecar_path,
    _get_tensor_path,
    _get_transition,
    _packed_span,
    get_binary_label,
)
from ragprobe.utils.logging import log


# The probability/uncertainty channels, in the canonical order the extractor writes them.
PROB_CHANNELS = ("p", "entropy", "surprisal", "top1_prob", "margin")


@dataclass
class ExampleSignals:
    """Compact signals for one example, enough to drive every Tier-A baseline."""

    example_id: str
    dataset: str
    split: str
    question: str
    answers: list
    transition: str | None
    label: float

    # Pooled probability/uncertainty stats over the answer tokens (training-free scalars).
    prob_pool: dict[str, float] = field(default_factory=dict)
    # Answer-span hidden representation (for SAPLMA): mean over answer tokens.
    answer_hidden_mean: np.ndarray | None = None
    # LLM-Check hidden score: log-det of the centered Gram matrix of answer hidden states.
    hidden_logdet: float = 0.0
    # Number of answer tokens actually present (diagnostics; >=2 needed for hidden_logdet).
    answer_len: int = 0


def _channel_index_by_name(item: dict) -> dict[str, int]:
    names = [str(n) for n in _get_feature_names(item)]
    return {name: i for i, name in enumerate(names)}


def _answer_slice(tensor: torch.Tensor, span: dict) -> torch.Tensor:
    start = int(span.get("start", 0) or 0)
    end = int(span.get("end", start) or start)
    return tensor[..., start:end]


def _pool_prob_channels(features: torch.Tensor, name_to_idx: dict[str, int], span: dict) -> dict[str, float]:
    """mean/min/max/last pooling of each probability channel over the answer tokens."""
    out: dict[str, float] = {}
    for ch in PROB_CHANNELS:
        idx = name_to_idx.get(ch)
        if idx is None:
            continue
        seg = _answer_slice(features[idx], span).float()
        if seg.numel() == 0:
            out[f"{ch}_mean"] = 0.0
            out[f"{ch}_min"] = 0.0
            out[f"{ch}_max"] = 0.0
            out[f"{ch}_last"] = 0.0
            continue
        out[f"{ch}_mean"] = float(seg.mean())
        out[f"{ch}_min"] = float(seg.min())
        out[f"{ch}_max"] = float(seg.max())
        out[f"{ch}_last"] = float(seg[-1])
    return out


def _hidden_gram_logdet(hidden: torch.Tensor) -> float:
    """LLM-Check 'Hidden Score': log-det of the centered token-covariance (EigenScore-style).

    hidden: [num_tokens, hidden_dim]. Uses the token Gram matrix (num_tokens x num_tokens)
    so it stays well-conditioned even though hidden_dim >> num_tokens.
    """
    h = hidden.float()
    n = h.shape[0]
    if n < 2:
        return 0.0
    hc = h - h.mean(dim=0, keepdim=True)
    gram = (hc @ hc.transpose(0, 1)) / float(hc.shape[1])
    gram = gram + 1e-3 * torch.eye(n, dtype=gram.dtype)
    sign, logabsdet = torch.linalg.slogdet(gram)
    if not torch.isfinite(logabsdet) or sign <= 0:
        return 0.0
    return float(logabsdet / n)


def _packed_row(item: dict):
    """(Pack, row) for this item when PROBE_PACKED_ROOT is set and the row is packed."""
    from ragprobe.training import packed_store

    if not packed_store.enabled():
        return None
    return packed_store._index().get(_get_tensor_path(item))


def _extract_one(item: dict, segment: str) -> ExampleSignals | None:
    example = _get_example(item)
    label = get_binary_label(
        item,
        positive_transition=item.get("_positive_transition", "NY"),
        label_mode=item.get("_label_mode", "without_context_correct"),
    )

    name_to_idx = _channel_index_by_name(item)
    span = _packed_span(item, segment)
    span_len = max(0, int(span.get("end", 0) or 0) - int(span.get("start", 0) or 0))

    tensor_path = _get_tensor_path(item)
    sidecar_path = _get_sidecar_path(item)
    have_raw = (tensor_path and os.path.exists(tensor_path)
                and sidecar_path and os.path.exists(sidecar_path))

    if have_raw:
        features = torch.load(tensor_path, map_location="cpu", weights_only=True)
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        prob_pool = _pool_prob_channels(features, name_to_idx, span)
        hidden = (sidecar.get("last_hidden_states") or {}).get(segment)
    else:
        # Raw tensors/sidecars deleted after packing: serve the identical values from
        # the packed store. Only valid when the pack's cap did not trim this span
        # (always true for segment="answer", cap == answer_tokens).
        hit = _packed_row(item)
        if hit is None:
            return None
        pack, row = hit
        seg = {"context": "c", "question": "q", "answer": "a"}[segment]
        feats_seg = pack.read(f"feats_{seg}", row, span_len)
        hidden = pack.read(f"hid_{seg}", row, span_len)
        if int(feats_seg.shape[-1]) != span_len or int(hidden.shape[0]) != span_len:
            return None  # cap-trimmed: packed values would diverge from the raw path
        prob_pool = _pool_prob_channels(
            feats_seg, name_to_idx, {"start": 0, "end": int(feats_seg.shape[-1])})
    answer_hidden_mean = None
    hidden_logdet = 0.0
    answer_len = 0
    if hidden is not None and hasattr(hidden, "numel") and hidden.numel() > 0:
        hidden = hidden.float()
        answer_len = int(hidden.shape[0])
        answer_hidden_mean = hidden.mean(dim=0).numpy().astype(np.float32)
        hidden_logdet = _hidden_gram_logdet(hidden)

    return ExampleSignals(
        example_id=str(example.get("id", "")),
        dataset=str(example.get("dataset", "")),
        split=str(example.get("split", "")),
        question=str(example.get("question", "")),
        answers=example.get("answers", example.get("answer", [])),
        transition=_get_transition(item),
        label=float(label),
        prob_pool=prob_pool,
        answer_hidden_mean=answer_hidden_mean,
        hidden_logdet=hidden_logdet,
        answer_len=answer_len,
    )


def extract_split_signals(
    meta_data: list[dict],
    segment: str = "answer",
    label_mode: str = "without_context_correct",
    positive_transition: str = "NY",
) -> list[ExampleSignals]:
    """Extract compact signals for every example in ``meta_data``.

    ``segment`` selects which token span / hidden state to read ("answer" for both the
    question-only and with-context settings, since correctness shows up on the answer
    tokens in both).
    """
    out: list[ExampleSignals] = []
    skipped = 0
    for item in meta_data:
        item["_label_mode"] = label_mode
        item["_positive_transition"] = positive_transition
        try:
            sig = _extract_one(item, segment)
        except Exception as exc:  # noqa: BLE001 - one bad example shouldn't kill the run
            sig = None
            log(f"baseline extract: skipping example due to {type(exc).__name__}: {exc}")
        if sig is None:
            skipped += 1
            continue
        out.append(sig)
    log(f"baseline extract: kept {len(out)} examples, skipped {skipped}, segment={segment}")
    return out
