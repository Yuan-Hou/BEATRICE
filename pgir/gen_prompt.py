"""Training-identical direct-answer prompt for the vLLM generation backend.

Spec §3.2: the prompt construction is IMPORTED from the service-side code
(`ragprobe.features.extractor` / `ragprobe.services.probe_scorer`), never
re-implemented here. If the service changes its prompt, this module follows
automatically; `tests/test_gen_prompt_source.py` statically asserts the
identity of every reused symbol and that no prompt text is defined here.

The token-id path (`prompt_token_ids`) reuses `_apply_chat_template` — the very
function the scorer's self-generation path calls — so the ids handed to vLLM
are byte-identical to what `ProbeScorer` would feed `model.generate`.
"""
from __future__ import annotations

from functools import lru_cache

# --- service-side code, reused verbatim (do NOT copy these implementations) ---
from ragprobe.features.extractor import (  # noqa: F401  (re-exported for tests)
    _apply_chat_template,
    _build_prompt,
    _format_context_legacy,
)
from ragprobe.services.probe_scorer import MODE_DEFAULTS, _normalize_contexts

# greedy answer budget must match the scorer's self-generation default
assert MODE_DEFAULTS["context"]["answer_tokens"] == MODE_DEFAULTS["question"]["answer_tokens"]
MAX_ANSWER_TOKENS: int = int(MODE_DEFAULTS["context"]["answer_tokens"])  # 32


def make_record(question: str, contexts=None, example_id: str | None = None) -> dict:
    """Build the record exactly like ProbeScorer._score_one does."""
    return {
        "example": {
            "id": example_id or "pgir",
            "question": str(question or "").strip(),
            "contexts": _normalize_contexts(contexts),
        }
    }


def resolve_mode(contexts) -> str:
    """Same auto-mode rule as ProbeScorer._resolve_mode (context iff contexts given)."""
    return "context" if contexts else "question"


def build_base_prompt(question: str, contexts=None) -> tuple[str, str]:
    """(mode, base_prompt) — the string the scorer's own generation path uses.

    Mirrors `_extract_one_unified`: question-mode gets QUESTION_ONLY_FEATURE_PROMPT,
    context-mode gets LEGACY_MUSIQUE_FEATURE_PROMPT over the '- '-joined contexts.
    """
    record = make_record(question, contexts)
    q = record["example"]["question"]
    if not q:
        raise ValueError("'question' is required")
    mode = resolve_mode(record["example"]["contexts"])
    context = None if mode == "question" else _format_context_legacy(record).strip()
    if mode == "context" and not context:
        raise ValueError("context mode but contexts rendered empty")
    _, base_prompt = _build_prompt(q, context)
    return mode, base_prompt


@lru_cache(maxsize=4)
def load_processor(model_name: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name)


def prompt_token_ids(processor, question: str, contexts=None) -> list[int]:
    """Chat-templated token ids for vLLM — via the service's `_apply_chat_template`."""
    _, base_prompt = build_base_prompt(question, contexts)
    enc = _apply_chat_template(processor, base_prompt, "cpu")
    return enc["input_ids"][0].tolist()
