"""TriviaQA (rc.wikipedia) as an OOD test set.

Unlike NQ, TriviaQA ships its own evidence: each question comes with Wikipedia
entity pages. Contexts are the leading portion of each entity page (truncated),
answers are the gold value + aliases. Streaming + shuffled so we never download
the full corpus.
"""
from __future__ import annotations

from itertools import islice

from datasets import load_dataset
from tqdm import tqdm

from .base import DatasetBuilder


class TriviaQABuilder(DatasetBuilder):
    name = "triviaqa"

    def __init__(
        self,
        split="validation",
        max_examples=None,
        max_pages=2,
        max_chars_per_page=1500,
        seed=20260622,
        **_,
    ):
        self.split = split
        self.max_examples = max_examples
        self.max_pages = int(max_pages)
        self.max_chars_per_page = int(max_chars_per_page)
        self.seed = seed

    def build(self):
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia",
                          split=self.split, streaming=True)
        ds = ds.shuffle(seed=self.seed, buffer_size=10000)
        it = islice(ds, int(self.max_examples)) if self.max_examples else ds

        records = []
        for i, item in enumerate(tqdm(it, total=self.max_examples, desc="triviaqa")):
            question = str(item.get("question") or "").strip()
            answer = item.get("answer") or {}
            value = str(answer.get("value") or "").strip()
            if not question or not value:
                continue
            aliases = [a for a in (answer.get("aliases") or []) if str(a).strip()]
            answers = list(dict.fromkeys([value] + aliases))
            pages = item.get("entity_pages") or {}
            titles = pages.get("title") or []
            texts = pages.get("wiki_context") or []
            contexts = []
            for rank, (t, txt) in enumerate(zip(titles, texts)):
                txt = str(txt or "").strip()
                if not txt:
                    continue
                contexts.append({
                    "text": txt[: self.max_chars_per_page],
                    "title": str(t), "source": "wikipedia",
                    "is_supporting": rank == 0, "metadata": {"rank": rank},
                })
                if len(contexts) >= self.max_pages:
                    break
            if not contexts:
                continue
            qid = str(item.get("question_id") or f"{i:06d}")
            records.append({
                "id": f"triviaqa-{self.split}-{qid}",
                "dataset": "triviaqa",
                "split": self.split,
                "question": question,
                "contexts": contexts,
                "answers": answers,
                "answerable": True,
                "metadata": {"question_id": qid},
            })
        records.sort(key=lambda r: r["id"])
        return records
