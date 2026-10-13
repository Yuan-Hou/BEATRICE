from datasets import load_dataset

from .base import DatasetBuilder


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


class MusiqueBuilder(DatasetBuilder):
    name = "musique"

    def __init__(self, include_splits=("train", "validation"), context_mode="supporting_only", max_examples=None):
        if context_mode not in {"supporting_only", "distractor_only"}:
            raise ValueError("context_mode must be one of: supporting_only, distractor_only")
        self.include_splits = include_splits
        self.context_mode = context_mode
        self.max_examples = max_examples

    def build(self):
        ds = load_dataset("bdsaglam/musique", "default")
        records = []
        count = 0
        for split in self.include_splits:
            for idx, item in enumerate(ds[split]):
                all_contexts = []
                for paragraph in item["paragraphs"]:
                    is_supporting = bool(paragraph.get("is_supporting"))
                    all_contexts.append({
                        "text": paragraph["paragraph_text"],
                        "title": paragraph.get("title"),
                        "source": "musique",
                        "is_supporting": is_supporting,
                        "metadata": paragraph,
                    })
                if self.context_mode == "supporting_only":
                    contexts = [ctx for ctx in all_contexts if ctx.get("is_supporting")]
                else:
                    contexts = _select_distractors_like_support_length(all_contexts)
                if not contexts:
                    continue
                answer = item.get("answer")
                answers = [answer] if isinstance(answer, str) and answer.strip() else []
                record_id = (
                    f"musique-{split}-{idx:06d}"
                    if self.context_mode == "supporting_only"
                    else f"musique-{self.context_mode}-{split}-{idx:06d}"
                )
                records.append({
                    "id": record_id,
                    "dataset": "musique",
                    "split": split,
                    "question": item["question"],
                    "contexts": contexts,
                    "answers": answers,
                    "answerable": bool(item.get("answerable", True)),
                    "metadata": {
                        "context_mode": self.context_mode,
                        "supporting_context_chars": sum(_text_len(ctx) for ctx in all_contexts if ctx.get("is_supporting")),
                        "selected_context_chars": sum(_text_len(ctx) for ctx in contexts),
                    },
                })
                count += 1
                if self.max_examples and count >= int(self.max_examples):
                    return records
        return records
