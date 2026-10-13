"""Natural Questions (nq_open) as an OOD test set, popqa-style.

nq_open provides question + gold answers only, so contexts are fetched live from
Wikipedia (same methodology as PopQABuilder): full-text search with the question,
top-k page intros become the retrieved contexts. This mirrors a realistic RAG
retrieval condition — contexts are relevant but not guaranteed answer-bearing.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .base import DatasetBuilder

WIKI_API = "https://en.wikipedia.org/w/api.php"


class NQBuilder(DatasetBuilder):
    name = "nq"

    def __init__(
        self,
        split="validation",
        max_examples=None,
        num_workers=16,
        contexts_per_question=2,
        wikipedia_user_agent=None,
        seed=20260622,
        **_,
    ):
        self.split = split
        self.max_examples = max_examples
        self.num_workers = num_workers
        self.contexts_per_question = int(contexts_per_question)
        self.seed = seed
        self.wikipedia_user_agent = wikipedia_user_agent or (
            "RAGProbeResearchBot/1.0 academic_research@example.com"
        )

    def build(self):
        ds = load_dataset("google-research-datasets/nq_open", split=self.split)
        ds = ds.shuffle(seed=self.seed)
        if self.max_examples:
            ds = ds.select(range(min(int(self.max_examples), len(ds))))
        session = self._build_session()

        def fetch(idx_item):
            idx, item = idx_item
            question = str(item["question"]).strip()
            if not question:
                return None
            contexts = self._search_contexts(session, question)
            if not contexts:
                return None
            answers = [a for a in (item.get("answer") or []) if str(a).strip()]
            if not answers:
                return None
            return {
                "id": f"nq-{self.split}-{idx:06d}",
                "dataset": "nq",
                "split": self.split,
                "question": question if question.endswith("?") else question + "?",
                "contexts": contexts,
                "answers": answers,
                "answerable": True,
                "metadata": {},
            }

        records = []
        indexed = list(enumerate(ds))
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(fetch, pair) for pair in indexed]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="nq"):
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
        records.sort(key=lambda r: r["id"])
        return records

    def _build_session(self):
        session = requests.Session()
        retry = Retry(total=4, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers.update({"User-Agent": self.wikipedia_user_agent})
        return session

    def _search_contexts(self, session, question: str):
        try:
            r = session.get(WIKI_API, params={
                "action": "query", "list": "search", "srsearch": question,
                "srlimit": self.contexts_per_question, "format": "json",
            }, timeout=20)
            titles = [hit["title"] for hit in r.json().get("query", {}).get("search", [])]
            if not titles:
                return []
            r = session.get(WIKI_API, params={
                "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                "titles": "|".join(titles), "format": "json",
            }, timeout=20)
            pages = r.json().get("query", {}).get("pages", {})
            by_title = {p.get("title"): (p.get("extract") or "").strip() for p in pages.values()}
            out = []
            for rank, t in enumerate(titles):
                text = by_title.get(t, "")
                if text:
                    out.append({"text": text[:2000], "title": t, "source": "wikipedia",
                                "is_supporting": rank == 0, "metadata": {"rank": rank}})
            return out
        except Exception:
            return []
