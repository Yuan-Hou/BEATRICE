"""STEER: probe-signal-driven choice of the next-evidence action.

`decide_steer` is a pure function over the arm pool — table-driven unit tests,
same discipline as `pipeline.decide`. It picks HOW the next evidence round is
fetched; `pipeline` executes the choice.

Rules (checked in order; motivation from the 2026 survey, see runs/RESULTS.md):
  1. lock-in guard — if the best context arm scores well BELOW the closed-book
     arm (delta), retrieval is hurting and the last answer is likely garbage;
     the default S2 query (question + last answer) would propagate that garbage
     into the next retrieval. Page fresh evidence with the original question
     instead ("s1"). [Retrieval-State Lock-In, arXiv 2606.22728: post-retrieval
     confidence is unreliable; the Q-only vs context branch disagreement is the
     signal that catches it.]
  2. stagnation — if per-round best context scores stopped moving (eps) while
     still below accept, the query formulation itself is exhausted; ask the LLM
     to rewrite it ("rewrite", costs 1 gen call).
  3. otherwise the default answer-guided query ("s2").

The rewrite prompt lives HERE, not in gen_prompt (gen_prompt must contain no
prompt text — tests/test_gen_prompt_source.py). Rewritten queries feed the
RETRIEVER only; answers are still generated with the training-format prompt.
"""
from __future__ import annotations

REWRITE_MAX_TOKENS = 64
MAX_WRONG_ANSWERS = 4

REWRITE_PROMPT = (
    "You are helping a search engine find evidence to answer a question.\n"
    "The answers found so far are likely wrong. Write ONE alternative search "
    "query that approaches the question differently (rephrase, decompose to a "
    "sub-question, or name the missing entity). Output ONLY the query, nothing "
    "else.\n\n"
    "Question: {question}\n"
    "Likely-wrong answers so far: {answers}\n"
    "Search query:"
)


def build_rewrite_prompt(question: str, wrong_answers: list[str]) -> str:
    seen, answers = set(), []
    for a in wrong_answers:
        key = " ".join(str(a).split()).casefold()
        if key and key not in seen:
            seen.add(key)
            answers.append(" ".join(str(a).split()))
    joined = "; ".join(answers[:MAX_WRONG_ANSWERS]) or "(none)"
    return REWRITE_PROMPT.format(question=str(question).strip(), answers=joined)


def clean_rewrite(text: str, question: str) -> str:
    """First non-empty line, de-quoted; empty if degenerate (caller falls back)."""
    line = next((ln.strip() for ln in str(text).splitlines() if ln.strip()), "")
    line = line.strip("\"'` ")
    if not line or line.casefold() == str(question).strip().casefold():
        return ""
    return line


def _round_best(arms) -> list[float]:
    """Per-round best p_cal of successful context arms, in round order.
    Robust to multiple arms per round (best-of-N)."""
    rounds: dict[int, float] = {}
    for a in arms:
        if a.status == "success" and a.mode == "context":
            r = a.round_index
            rounds[r] = max(rounds.get(r, float("-inf")), a.p_cal)
    return [rounds[r] for r in sorted(rounds)]


def decide_steer(pool, cfg) -> str:
    """-> "s1" | "rewrite" | "s2" (default). Pure; no I/O."""
    q_best = max((a.p_cal for a in pool
                  if a.status == "success" and a.mode == "question"),
                 default=None)
    ctx_seq = _round_best(pool)
    if not ctx_seq:
        return "s2"  # no context round yet: nothing to steer on
    if q_best is not None and ctx_seq[-1] < q_best - cfg.steer_delta:
        return "s1"
    if len(ctx_seq) >= 2 and abs(ctx_seq[-1] - ctx_seq[-2]) < cfg.steer_eps \
            and ctx_seq[-1] < cfg.tau_hi:
        return "rewrite"
    return "s2"
