"""next_evidence strategies + anti-lock-in guard (spec §4.3).

S1 deepen  — same query, next page (offset emulated by over-fetching + slicing;
             FlashRAG retrievers have no native offset).
S2 rewrite — answer-guided query: question + " " + last_answer (default).
Guard      — if this round's evidence pid-set Jaccard-overlaps the previous
             round's by > 0.6, force one S1 page instead of S2.

S3 (probe-guided evidence re-ranking) is a later ablation — not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

OVERLAP_GUARD = 0.6
PLATEAU_EPS = 0.02
ANSWER_QUERY_TOKENS = 32  # whitespace tokens; answers are <=32 model tokens anyway


def render_contexts(docs: list[dict]) -> list[dict]:
    """Retrieved docs -> probe_scorer `contexts` payload."""
    out = []
    for d in docs:
        text = d.get("contents") or d.get("text") or d.get("paragraph_text") or ""
        out.append({"text": str(text)})
    return out


def _doc_id(d: dict):
    return d.get("id", d.get("doc_id", d.get("contents", "")[:80]))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


@dataclass
class EvidenceState:
    used_pids: set = field(default_factory=set)
    round_pids: list = field(default_factory=list)   # one pid-set per evidence round
    pages: dict = field(default_factory=dict)        # query -> next page index
    forced_s1: int = 0
    overlaps: list = field(default_factory=list)

    def register(self, docs: list[dict]) -> None:
        pids = {_doc_id(d) for d in docs}
        if self.round_pids:
            self.overlaps.append(jaccard(pids, self.round_pids[-1]))
        self.round_pids.append(pids)
        self.used_pids |= pids

    def diagnostics(self, pool) -> dict:
        ctx = [a for a in pool if a.mode == "context" and a.status == "success"]
        deltas = [abs(b.p_cal - a.p_cal) for a, b in zip(ctx, ctx[1:])]
        return {
            "evidence_overlap_rate": (sum(self.overlaps) / len(self.overlaps)
                                      if self.overlaps else 0.0),
            "overlaps": list(self.overlaps),
            "forced_s1": self.forced_s1,
            "score_plateau_rate": (sum(1 for d in deltas if d < PLATEAU_EPS) / len(deltas)
                                   if deltas else 0.0),
        }


def _truncate_ws(text: str, n_tokens: int) -> str:
    return " ".join(str(text).split()[:n_tokens])


def _page(question: str, retriever, state: EvidenceState, k: int) -> list[dict]:
    """S1: next page of the original query, skipping already-used pids."""
    page = state.pages.get(question, 1)
    state.pages[question] = page + 1
    fetched = retriever.search(question, k * (page + 1))
    fresh = [d for d in fetched if _doc_id(d) not in state.used_pids]
    return fresh[:k] if fresh else fetched[-k:]


def s3_candidates(question: str, pool, retriever, state: EvidenceState, cfg,
                  m: int = 3) -> tuple[list[list[dict]], int]:
    """S3 ablation (spec §4.3, default off): m candidate evidence sets from one
    over-fetched S2 query; the caller evaluates each and keeps the best arm.
    Cost is m arms per round — reported as-is on the Pareto plot."""
    k = cfg.k
    live = [a for a in pool if a["status" if isinstance(a, dict) else "status"] == "success"] \
        if pool and isinstance(pool[0], dict) else [a for a in pool if a.status == "success"]
    last_answer = (live[-1]["answer"] if isinstance(live[-1], dict) else live[-1].answer) if live else ""
    query = (question + " " + _truncate_ws(last_answer, ANSWER_QUERY_TOKENS)).strip()
    fetched = retriever.search(query, k * m + len(state.used_pids))
    fresh = [d for d in fetched if _doc_id(d) not in state.used_pids]
    sets = [fresh[j * k:(j + 1) * k] for j in range(m)]
    sets = [s for s in sets if s]
    if not sets:
        sets = [_page(question, retriever, state, k)]
        return sets, 2
    return sets, 1


def next_evidence(question: str, pool, docs: list[dict], retriever,
                  state: EvidenceState, cfg, strategy: str = "s2",
                  override_query: str | None = None) -> tuple[list[dict], int]:
    """Returns (new_docs, n_retriever_calls). Registers the new round in state.

    override_query (STEER rewrite) replaces the S2 answer-guided query and
    skips the overlap guard — the rewritten query IS the anti-lock-in action."""
    k = cfg.k
    n_calls = 0
    guard = (bool(state.overlaps) and state.overlaps[-1] > OVERLAP_GUARD
             and not override_query)

    new_docs: list[dict] = []
    if strategy == "s1" or guard:
        if guard and strategy != "s1":
            state.forced_s1 += 1
        new_docs = _page(question, retriever, state, k)
        n_calls += 1
    else:  # S2 answer-guided rewrite (or STEER-rewritten query)
        if override_query:
            query = str(override_query).strip()
        else:
            live = [a for a in pool if a.status == "success"]
            last_answer = live[-1].answer if live else ""
            query = (question + " " + _truncate_ws(last_answer, ANSWER_QUERY_TOKENS)).strip()
        fetched = retriever.search(query, k + len(state.used_pids))
        n_calls += 1
        fresh = [d for d in fetched if _doc_id(d) not in state.used_pids]
        new_docs = fresh[:k]
        if not new_docs:  # S2 exhausted -> deepen instead
            new_docs = _page(question, retriever, state, k)
            n_calls += 1

    state.register(new_docs)
    return new_docs, n_calls
