from datasets import load_dataset

from .base import DatasetBuilder


def _paragraph_text(title, sentences):
    return title + "\n" + "".join(sentences)


def _supporting_fact_map(item):
    sf = item.get("supporting_facts", {})
    titles = sf.get("title", []) or []
    sent_ids = sf.get("sent_id", []) or []

    out = {}
    for title, sent_id in zip(titles, sent_ids):
        out.setdefault(title, set()).add(int(sent_id))
    return out


def _text_len(ctx):
    return len(str(ctx.get("text") or ""))


def _select_distractors_like_support_length(contexts):
    supporting = [ctx for ctx in contexts if ctx.get("is_supporting")]
    distractors = [ctx for ctx in contexts if not ctx.get("is_supporting")]
    if not distractors:
        return []
    target_chars = sum(_text_len(ctx) for ctx in supporting)
    if target_chars <= 0:
        return []
    selected = []
    total = 0
    for ctx in distractors:
        selected.append(ctx)
        total += _text_len(ctx)
        if total >= target_chars:
            break
    return selected


class HotpotQABuilder(DatasetBuilder):
    """HotpotQA oracle-RAG adapter.

    Supported context modes:
    - supporting_only: only keep paragraphs whose title appears in supporting_facts.
    - distractor_only: keep non-supporting paragraphs until total length roughly matches supporting paragraphs.
    """

    name = "hotpotqa"

    def __init__(
        self,
        config_name="distractor",
        include_splits=("train", "validation"),
        context_mode="supporting_only",
        max_examples=None,
    ):
        if context_mode not in {"supporting_only", "distractor_only"}:
            raise ValueError("context_mode must be one of: supporting_only, distractor_only")
        self.config_name = config_name
        self.include_splits = include_splits
        self.context_mode = context_mode
        self.max_examples = max_examples

    def build(self):
        ds = load_dataset("hotpotqa/hotpot_qa", self.config_name)
        records = []
        count = 0
        for split in self.include_splits:
            if split not in ds:
                continue
            for idx, item in enumerate(ds[split]):
                support_map = _supporting_fact_map(item)
                support_titles = set(support_map.keys())
                titles = item["context"]["title"]
                paragraphs = item["context"]["sentences"]
                all_contexts = []
                for title, sentences in zip(titles, paragraphs):
                    is_supporting = title in support_titles
                    supporting_sentence_ids = sorted(list(support_map.get(title, set())))
                    supporting_sentences = [sentences[i] for i in supporting_sentence_ids if 0 <= i < len(sentences)]
                    all_contexts.append({
                        "text": _paragraph_text(title, sentences),
                        "title": title,
                        "source": "hotpotqa",
                        "is_supporting": is_supporting,
                        "metadata": {
                            "supporting_sentence_ids": supporting_sentence_ids,
                            "supporting_sentences": supporting_sentences,
                            "num_sentences": len(sentences),
                        },
                    })
                if self.context_mode == "supporting_only":
                    contexts = [ctx for ctx in all_contexts if ctx.get("is_supporting")]
                else:
                    contexts = _select_distractors_like_support_length(all_contexts)
                if not contexts:
                    continue
                records.append({
                    "id": f"hotpotqa-{self.context_mode}-{split}-{item.get('id', idx)}",
                    "dataset": "hotpotqa",
                    "split": split,
                    "question": item["question"],
                    "contexts": contexts,
                    "answers": [str(item["answer"]).strip()],
                    "answerable": True,
                    "metadata": {
                        "original_id": item.get("id"),
                        "type": item.get("type"),
                        "level": item.get("level"),
                        "supporting_facts": item.get("supporting_facts"),
                        "context_mode": self.context_mode,
                        "supporting_context_chars": sum(_text_len(ctx) for ctx in all_contexts if ctx.get("is_supporting")),
                        "selected_context_chars": sum(_text_len(ctx) for ctx in contexts),
                    },
                })
                count += 1
                if self.max_examples and count >= int(self.max_examples):
                    return records
        return records
