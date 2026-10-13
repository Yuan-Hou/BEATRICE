import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ragprobe.prompts.qa import format_context
from ragprobe.training import packed_store
from ragprobe.utils.io import read_jsonl

# Intra-batch parallel fetch (PROBE_FETCH_THREADS>1): DataLoader(num_workers=0)
# calls __getitems__ per batch; a thread pool over __getitem__ keeps sampler
# order, RNG and collate identical to the serial path — pure IO/CPU overlap.
# (num_workers>0 stays banned: 8GB batches through /dev/shm OOM the host.)
_FETCH_THREADS = int(os.environ.get("PROBE_FETCH_THREADS", "0"))
_fetch_pool = None


def _parallel_getitems(dataset, indices):
    global _fetch_pool
    if _FETCH_THREADS <= 1:
        return [dataset[i] for i in indices]
    if _fetch_pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _fetch_pool = ThreadPoolExecutor(max_workers=_FETCH_THREADS)
    return list(_fetch_pool.map(dataset.__getitem__, indices))


def fast_collate(rows):
    # packed batch-direct mode: __getitems__ already returned a collated batch tuple
    if isinstance(rows, tuple):
        return rows
    """Bit-identical replacement for default_collate on tuples of fixed-shape
    tensors: torch.stack is a single-threaded memcpy (~2.4s of a 4GB ctx batch);
    threaded row-wise copy_ into a preallocated tensor uses the idle cores.
    Any field that isn't a uniform-shape tensor falls back to default_collate."""
    from torch.utils.data._utils.collate import default_collate
    first = rows[0]
    if not isinstance(first, (tuple, list)):
        return default_collate(rows)
    out = []
    for f in range(len(first)):
        col = [r[f] for r in rows]
        v = col[0]
        if (torch.is_tensor(v)
                and all(x.shape == v.shape and x.dtype == v.dtype for x in col)):
            dst = torch.empty((len(col),) + tuple(v.shape), dtype=v.dtype)
            if _FETCH_THREADS > 1 and dst.numel() * dst.element_size() > (1 << 24):
                global _fetch_pool
                if _fetch_pool is None:
                    from concurrent.futures import ThreadPoolExecutor
                    _fetch_pool = ThreadPoolExecutor(max_workers=_FETCH_THREADS)
                list(_fetch_pool.map(lambda i: dst[i].copy_(col[i]), range(len(col))))
            else:
                torch.stack(col, out=dst)
            out.append(dst)
        else:
            out.append(default_collate(col))
    return out


RAG_SIGNAL_CLASSES = (
    "Attention Bias",
    "Logic Drift",
    "Hierarchy Misalignment",
    "Over-correction",
    "Others",
    "Not Regression",
)
RAG_SIGNAL_CLASS_TO_ID = {name: i for i, name in enumerate(RAG_SIGNAL_CLASSES)}
RAG_CLASS_IGNORE_INDEX = -100


def _get_example(item: dict) -> dict:
    """Return canonical example for both new-schema and legacy meta records."""
    if "example" in item:
        return item["example"]

    # Legacy schema from your old scripts:
    # question, context, answer, answerable, set
    raw_contexts = item.get("contexts", item.get("context", []))
    contexts = []
    for i, ctx in enumerate(raw_contexts):
        if isinstance(ctx, dict):
            contexts.append(ctx)
        else:
            contexts.append({
                "text": str(ctx),
                "title": None,
                "source": "legacy",
                "is_supporting": True,
                "metadata": {"legacy_index": i},
            })

    raw_answer = item.get("answers", item.get("answer", []))
    if isinstance(raw_answer, list):
        answers = raw_answer
    elif raw_answer is None:
        answers = []
    else:
        answers = [str(raw_answer)]

    return {
        "id": str(item.get("id", item.get("question", ""))),
        "dataset": item.get("dataset", "legacy"),
        "split": item.get("split", item.get("set", "unknown")),
        "question": item["question"],
        "contexts": contexts,
        "answers": answers,
        "answerable": item.get("answerable", True),
        "metadata": item.get("metadata", {}),
    }


def _get_split(item: dict) -> str:
    return _get_example(item).get("split", item.get("set", "unknown"))


def _get_transition(item: dict) -> Optional[str]:
    """Return NY/YN/YY/NN for both new-schema and legacy records."""
    transition = item.get("derived", {}).get("transition")
    if transition:
        return transition

    if "is_correct_without_context" in item or "is_correct_with_context" in item:
        if item["is_correct_without_context"] and item["is_correct_with_context"]:
            base_corr = bool(item.get("is_correct_without_context", False))
            rag_corr = bool(item.get("is_correct_with_context", False))
            return ("Y" if base_corr else "N") + ("Y" if rag_corr else "N")
        else:
            return None

    return None


def _get_tensor_path(item: dict) -> Optional[str]:
    # New schema
    path = item.get("feature_extraction", {}).get("rag", {}).get("tensor_path")
    if path:
        return path

    # Legacy / patched schema fallback
    path = item.get("tensor_path")
    if path:
        return os.path.join("legacy", path)

    path = item.get("rag_tensor_path")
    if path:
        return path

    return None


def _get_token_ids_path(item: dict) -> Optional[str]:
    path = item.get("feature_extraction", {}).get("rag", {}).get("token_ids_path")
    if path:
        return path
    return item.get("token_ids_path")


def _get_feature_names(item: dict) -> list:
    return item.get("feature_extraction", {}).get("rag", {}).get("feature_names", []) or []


def _get_feature_target_type(item: dict) -> str:
    return item.get("feature_extraction", {}).get("rag", {}).get("target_type", "contexts")


def _is_valid_feature_record(item: dict) -> bool:
    # New schema
    fe = item.get("feature_extraction", {})
    if fe.get("status") != "success" or _get_tensor_path(item) is None:
        return False

    if _get_transition(item) not in {"NY", "YN", "YY", "NN"}:
        return False
    return True



_PARTITION_ID_CACHE = {}


def _load_partition_ids(path: str, group: str) -> set[str]:
    cache_key = (str(path), str(group))
    if cache_key in _PARTITION_ID_CACHE:
        return _PARTITION_ID_CACHE[cache_key]
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = set()
    if isinstance(data, dict):
        if isinstance(data.get("ids"), dict) and data["ids"].get(group) is not None:
            ids.update(str(x) for x in data["ids"].get(group) or [])
        for ds_data in (data.get("by_dataset") or {}).values():
            values = ds_data.get(f"{group}_ids") if isinstance(ds_data, dict) else None
            if values is not None:
                ids.update(str(x) for x in values)
    _PARTITION_ID_CACHE[cache_key] = ids
    return ids


def _partition_keep(item: dict, partition: Optional[dict]) -> bool:
    if not partition or not bool(partition.get("enabled", True)):
        return True
    source_splits = partition.get("source_splits")
    if source_splits is not None and _get_split(item) not in set(source_splits):
        return True
    ids_path = partition.get("ids_path")
    group = partition.get("group")
    if ids_path and group:
        return (_get_example(item).get("id") or "") in _load_partition_ids(ids_path, group)
    return True

def load_feature_meta(paths: List[str], splits=None, partition: Optional[dict] = None):
    """Load feature meta records with compatibility for old and new schemas.

    New schema expected by the refactored project:
      item["example"]["split"], item["derived"]["transition"],
      item["feature_extraction"]["rag"]["tensor_path"]

    Legacy schema from your old scripts:
      item["set"], item["is_correct_without_context"],
      item["is_correct_with_context"], item["feature_extraction"]["rag"]["tensor_path"]
    """
    splits = set(splits) if splits else None
    records = []

    missing_paths = []
    total_seen = 0
    split_filtered = 0
    partition_filtered = 0
    invalid_feature = 0

    for path in paths:
        if not os.path.exists(path):
            missing_paths.append(path)
            continue

        for item in read_jsonl(path):
            total_seen += 1

            if splits is not None and _get_split(item) not in splits:
                split_filtered += 1
                continue

            if not _partition_keep(item, partition):
                partition_filtered += 1
                continue

            if not _is_valid_feature_record(item):
                invalid_feature += 1
                continue

            records.append(item)

    if missing_paths:
        print("[load_feature_meta] Missing paths:")
        for p in missing_paths:
            print(f"  - {p}")

    print(
        "[load_feature_meta] "
        f"seen={total_seen}, loaded={len(records)}, "
        f"split_filtered={split_filtered}, partition_filtered={partition_filtered}, invalid_feature={invalid_feature}, "
        f"splits={sorted(splits) if splits else None}, partition={partition}"
    )

    return records


def get_without_context_correct(item: dict) -> Optional[bool]:
    runs = item.get("runs", {})
    if isinstance(runs, dict) and isinstance(runs.get("without_context"), dict):
        value = runs["without_context"].get("is_correct")
        if value is not None:
            return bool(value)
    if "is_correct_without_context" in item:
        return bool(item.get("is_correct_without_context"))
    transition = _get_transition(item)
    if transition in {"YN", "YY"}:
        return True
    if transition in {"NN", "NY"}:
        return False
    return None


def get_with_context_correct(item: dict) -> Optional[bool]:
    runs = item.get("runs", {})
    if isinstance(runs, dict) and isinstance(runs.get("with_context"), dict):
        value = runs["with_context"].get("is_correct")
        if value is not None:
            return bool(value)
    if "is_correct_with_context" in item:
        return bool(item.get("is_correct_with_context"))
    transition = _get_transition(item)
    if transition in {"NY", "YY"}:
        return True
    if transition in {"NN", "YN"}:
        return False
    return None


def get_rag_harmful(item: dict) -> bool:
    transition = _get_transition(item)
    return transition == "YN"


def get_aux_binary_labels(item: dict) -> list[float]:
    with_context_correct = get_with_context_correct(item)
    return [
        1.0 if with_context_correct else 0.0,
        1.0 if get_rag_harmful(item) else 0.0,
    ]


def get_binary_label(item: dict, positive_transition="NY", label_mode="transition_positive") -> float:
    if label_mode == "without_context_correct":
        value = get_without_context_correct(item)
        return 1.0 if value else 0.0
    if label_mode == "with_context_correct":
        value = get_with_context_correct(item)
        return 1.0 if value else 0.0
    return get_helpful_label(item, positive_transition=positive_transition)


def get_helpful_label(item: dict, positive_transition="NY") -> float:
    transition = _get_transition(item)
    return 1.0 if transition == positive_transition else 0.0


def _get_example_id(item: dict) -> str:
    return str(_get_example(item).get("id", ""))


def normalize_rag_signal_category(category) -> Optional[str]:
    if category is None:
        return None

    raw = str(category).strip().lower()
    if raw.startswith("at"):
        return "Attention Bias"
    if raw.startswith("lo"):
        return "Logic Drift"
    if raw.startswith("hi"):
        return "Hierarchy Misalignment"
    if raw.startswith("ov"):
        return "Over-correction"
    if raw.startswith("other"):
        return "Others"
    if raw.startswith("not") or raw.startswith("non"):
        return "Not Regression"
    return None


def get_rag_signal_label(item: dict, regression_transition="YN") -> int:
    """Return the 6-way auxiliary RAG signal label, or ignore_index if unknown."""
    transition = _get_transition(item)
    if transition is None:
        return RAG_CLASS_IGNORE_INDEX

    if transition != regression_transition:
        return RAG_SIGNAL_CLASS_TO_ID["Not Regression"]

    analysis = item.get("regression_analysis")
    if not analysis:
        return RAG_CLASS_IGNORE_INDEX

    category = normalize_rag_signal_category(analysis.get("error_category"))
    if category is None:
        return RAG_CLASS_IGNORE_INDEX

    return RAG_SIGNAL_CLASS_TO_ID[category]


def load_rag_signal_label_map(paths: List[str], regression_transition="YN"):
    labels = {}
    missing_paths = []

    for path in paths or []:
        if not os.path.exists(path):
            missing_paths.append(path)
            continue

        for item in read_jsonl(path):
            label = get_rag_signal_label(item, regression_transition=regression_transition)
            if label == RAG_CLASS_IGNORE_INDEX:
                continue

            example_id = _get_example_id(item)
            if example_id:
                labels[example_id] = label

    if missing_paths:
        print("[load_rag_signal_label_map] Missing paths:")
        for p in missing_paths:
            print(f"  - {p}")

    print(
        "[load_rag_signal_label_map] "
        f"loaded_labels={len(labels)}, regression_transition={regression_transition}"
    )
    return labels


class RAGFeatureDataset(Dataset):
    def __init__(
        self,
        meta_data,
        tokenizer,
        max_len=1024,
        positive_transition="NY",
        label_mode="transition_positive",
        rag_class_labels_by_id=None,
        regression_transition="YN",
        use_numeric_features=True,
        use_attention_features=True,
        use_last_hidden=True,
    ):
        self.meta_data = meta_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.positive_transition = positive_transition
        self.label_mode = label_mode
        self.rag_class_labels_by_id = rag_class_labels_by_id or {}
        self.regression_transition = regression_transition
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        item = self.meta_data[idx]
        example = _get_example(item)
        tensor_path = _get_tensor_path(item)

        token_ids_path = _get_token_ids_path(item)
        if token_ids_path and os.path.exists(token_ids_path):
            token_ids_tensor = torch.load(token_ids_path, map_location="cpu", weights_only=True).to(torch.long)
            if token_ids_tensor.numel() > self.max_len:
                token_ids_tensor = token_ids_tensor[:self.max_len]
            elif token_ids_tensor.numel() < self.max_len:
                token_ids_tensor = F.pad(token_ids_tensor, (0, self.max_len - token_ids_tensor.numel()), value=self.pad_id)
        else:
            if _get_feature_target_type(item) == "question":
                token_text = example["question"].strip()
            else:
                token_text = format_context(example["contexts"])
            token_ids = self.tokenizer(token_text, add_special_tokens=False)["input_ids"]
            if len(token_ids) > self.max_len:
                token_ids = token_ids[:self.max_len]
            else:
                token_ids = token_ids + [self.pad_id] * (self.max_len - len(token_ids))
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

        label = get_binary_label(item, self.positive_transition, self.label_mode)
        aux_binary_labels = get_aux_binary_labels(item)
        rag_class_label = self.rag_class_labels_by_id.get(
            _get_example_id(item),
            get_rag_signal_label(item, regression_transition=self.regression_transition),
        )

        if not tensor_path or not os.path.exists(tensor_path):
            features = torch.zeros((131, self.max_len), dtype=torch.float32)
        else:
            features = torch.load(tensor_path, map_location="cpu", weights_only=True).to(torch.float32)
            c_len = features.shape[1]
            if c_len < self.max_len:
                features = F.pad(features, (0, self.max_len - c_len))
            else:
                features = features[:, :self.max_len]

            feature_names = _get_feature_names(item)
            has_surprisal = any(str(name).endswith("surprisal") or str(name) == "surprisal" for name in feature_names)
            if not has_surprisal:
                # Legacy extractor stored [attention_channels + p_value + entropy].
                # New extractors include surprisal explicitly.
                p_index = -2
                if feature_names:
                    for i, name in enumerate(feature_names):
                        if str(name).endswith("p") or str(name) in {"p", "p_value"}:
                            p_index = i
                            break
                p_values = features[p_index:p_index + 1, :].clamp_min(1e-10)
                surprisal = -torch.log(p_values)
                features = torch.cat([features, surprisal], dim=0)

        return (
            features,
            token_ids_tensor,
            torch.tensor([label], dtype=torch.float32),
            torch.tensor(rag_class_label, dtype=torch.long),
            torch.tensor(aux_binary_labels, dtype=torch.float32),
            idx,
        )


def _get_sidecar_path(item: dict) -> Optional[str]:
    return item.get("feature_extraction", {}).get("rag", {}).get("sidecar_path")


def _packed_span(item: dict, name: str) -> dict:
    return item.get("feature_extraction", {}).get("rag", {}).get("spans", {}).get("packed", {}).get(name, {}) or {}


def _feature_channel_indices(item: dict, prefixes: tuple[str, ...]) -> list[int]:
    names = _get_feature_names(item)
    out = []
    for i, name in enumerate(names):
        s = str(name)
        if any(s == p or s.startswith(p) for p in prefixes):
            out.append(i)
    return out


def _segment_slice(tensor: torch.Tensor, span: dict, max_len: int, fill_value=0):
    start = int(span.get("start", 0) or 0)
    end = int(span.get("end", start) or start)
    part = tensor[..., start:end]
    length = int(part.shape[-1])
    if length > max_len:
        part = part[..., :max_len]
    elif length < max_len:
        part = F.pad(part, (0, max_len - length), value=fill_value)
    return part


def _attention_entropy(values: torch.Tensor, dim: int) -> torch.Tensor:
    values = values.float().clamp_min(0.0)
    denom = values.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    probs = values / denom
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=dim)
    return ent


class QuestionAnswerSidecarDataset(Dataset):
    """Dataset for question-only sidecar features.

    It builds two token-centered tensors:
      - question branch: probability/residual channels over question tokens plus
        entropy across answer-token attention for each layer/head.
      - answer branch: probability/residual channels over answer tokens plus
        entropy across question-token attention for each layer/head.
    """

    def __init__(
        self,
        meta_data,
        tokenizer,
        question_max_len=128,
        answer_max_len=32,
        positive_transition="NY",
        label_mode="without_context_correct",
        use_numeric_features=True,
        use_probability_features=True,
        use_residual_features=True,
        use_attention_features=True,
        use_last_hidden=True,
        use_attention_mass=False,
        ablate_segments=(),
        label_override=None,
        label_override_required=False,
    ):
        if label_override_required and label_override is not None:
            # protocol-faithful unanimity filtering: rows without an aligned
            # label (judge not unanimous) are excluded from the dataset
            meta_data = [m for m in meta_data if _get_example_id(m) in label_override]
        self.meta_data = meta_data
        self.tokenizer = tokenizer
        self.question_max_len = int(question_max_len)
        self.answer_max_len = int(answer_max_len)
        self.positive_transition = positive_transition
        self.label_mode = label_mode
        self.use_numeric_features = bool(use_numeric_features)
        self.use_probability_features = bool(use_probability_features)
        self.use_residual_features = bool(use_residual_features)
        self.use_attention_features = bool(use_attention_features)
        self.use_last_hidden = bool(use_last_hidden)
        self.use_attention_mass = bool(use_attention_mass)
        # ablate_segments: subset of {"question","answer"} -> zero that segment's
        # numeric/hidden/attention features (masks and dims kept, same convention as
        # the with-context dataset's hook).
        self.ablate_segments = frozenset(ablate_segments or ())
        # {example_id -> 0.0/1.0}: replaces get_binary_label when the id is present
        # (aligned-label experiments; base/rewritten disambiguation is the ctx
        # dataset's job via rewritten_label_override)
        self.label_override = label_override
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        prefixes = []
        if self.use_probability_features:
            prefixes.extend(["p", "entropy", "surprisal", "top1_prob", "margin"])
        if self.use_residual_features:
            prefixes.append("resid_")
        self.numeric_prefixes = tuple(prefixes)
        # packed fast path (PROBE_PACKED_ROOT): all-or-nothing per dataset, see packed_store
        self._packed = packed_store.resolve(self.meta_data) if packed_store.enabled() else None

    def __len__(self):
        return len(self.meta_data)

    def __getitems__(self, indices):
        # batch-direct packed mode: opted in per instance by the trainer's
        # fast_collate sites (fast_collate passes the ready batch through)
        if self._packed is not None and getattr(self, "_packed_batch", False):
            return packed_store.build_question_batch(self, indices)
        return _parallel_getitems(self, indices)

    def __getitem__(self, idx):
        item = self.meta_data[idx]
        if self._packed is not None:
            pack, row = self._packed[idx]
            return packed_store.build_question_item(self, item, pack, row, idx)
        tensor_path = _get_tensor_path(item)
        token_ids_path = _get_token_ids_path(item)
        sidecar_path = _get_sidecar_path(item)
        if not tensor_path or not os.path.exists(tensor_path):
            raise FileNotFoundError(f"missing tensor_path for idx={idx}: {tensor_path}")
        if not token_ids_path or not os.path.exists(token_ids_path):
            raise FileNotFoundError(f"missing token_ids_path for idx={idx}: {token_ids_path}")
        if not sidecar_path or not os.path.exists(sidecar_path):
            raise FileNotFoundError(f"missing sidecar_path for idx={idx}: {sidecar_path}")

        features = torch.load(tensor_path, map_location="cpu", weights_only=True).to(torch.float32)
        token_ids = torch.load(token_ids_path, map_location="cpu", weights_only=True).to(torch.long)
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)

        q_span = _packed_span(item, "question")
        a_span = _packed_span(item, "answer")
        numeric_idx = _feature_channel_indices(item, self.numeric_prefixes)
        if not numeric_idx:
            numeric_idx = list(range(min(5, features.shape[0]))) if self.use_probability_features else [0]
        numeric = features[numeric_idx]

        q_numeric = _segment_slice(numeric, q_span, self.question_max_len)
        a_numeric = _segment_slice(numeric, a_span, self.answer_max_len)
        if not self.use_numeric_features:
            q_numeric = torch.zeros_like(q_numeric)
            a_numeric = torch.zeros_like(a_numeric)
        q_token_ids = _segment_slice(token_ids, q_span, self.question_max_len, fill_value=self.pad_id).to(torch.long)
        a_token_ids = _segment_slice(token_ids, a_span, self.answer_max_len, fill_value=self.pad_id).to(torch.long)
        q_hidden = _hidden_segment(sidecar, "question", self.question_max_len)
        a_hidden = _hidden_segment(sidecar, "answer", self.answer_max_len)
        if not self.use_last_hidden:
            q_hidden = torch.zeros_like(q_hidden)
            a_hidden = torch.zeros_like(a_hidden)

        q_len = min(int(q_span.get("len", 0) or 0), self.question_max_len)
        a_len = min(int(a_span.get("len", 0) or 0), self.answer_max_len)
        q_mask = torch.zeros(self.question_max_len, dtype=torch.bool)
        a_mask = torch.zeros(self.answer_max_len, dtype=torch.bool)
        q_mask[:q_len] = True
        a_mask[:a_len] = True

        aq = ((sidecar.get("attention_maps") or {}).get("answer_to_question"))
        if aq is None or aq.numel() == 0:
            q_attn = torch.zeros((0, self.question_max_len), dtype=torch.float32)
            a_attn = torch.zeros((0, self.answer_max_len), dtype=torch.float32)
        else:
            aq = aq.float()[:, :, :a_len, :q_len]
            if q_len > 0 and a_len > 0:
                # entropy -- how focused the answer<->question attention is
                q_ent = _attention_entropy(aq, dim=2).reshape(-1, q_len)
                a_ent = _attention_entropy(aq, dim=3).reshape(-1, a_len)
                if self.use_attention_mass:
                    # + normalized segment-mass simplex (since each query row sums to 1
                    # over all keys, summing the question-key block IS that segment's
                    # share). a_mass = answer token's mass on the question (1-that=self);
                    # q_mass = attention each question token receives. Symmetric channels.
                    q_mass = aq.sum(dim=2).reshape(-1, q_len)
                    a_mass = aq.sum(dim=3).reshape(-1, a_len)
                    q_attn_raw = torch.cat([q_ent, q_mass], dim=0)
                    a_attn_raw = torch.cat([a_ent, a_mass], dim=0)
                else:
                    q_attn_raw, a_attn_raw = q_ent, a_ent
            else:
                mult = 2 if self.use_attention_mass else 1
                q_attn_raw = torch.zeros((aq.shape[0] * aq.shape[1] * mult, 0))
                a_attn_raw = torch.zeros((aq.shape[0] * aq.shape[1] * mult, 0))
            q_attn = F.pad(q_attn_raw, (0, self.question_max_len - q_attn_raw.shape[-1])) if q_attn_raw.shape[-1] < self.question_max_len else q_attn_raw[:, :self.question_max_len]
            a_attn = F.pad(a_attn_raw, (0, self.answer_max_len - a_attn_raw.shape[-1])) if a_attn_raw.shape[-1] < self.answer_max_len else a_attn_raw[:, :self.answer_max_len]
        if not self.use_attention_features:
            q_attn = torch.zeros_like(q_attn)
            a_attn = torch.zeros_like(a_attn)
        if "question" in self.ablate_segments:
            q_numeric = torch.zeros_like(q_numeric)
            q_hidden = torch.zeros_like(q_hidden)
            q_attn = torch.zeros_like(q_attn)
        if "answer" in self.ablate_segments:
            a_numeric = torch.zeros_like(a_numeric)
            a_hidden = torch.zeros_like(a_hidden)
            a_attn = torch.zeros_like(a_attn)

        label = get_binary_label(item, self.positive_transition, self.label_mode)
        if self.label_override is not None:
            ov = self.label_override.get(_get_example_id(item))
            if ov is not None:
                label = float(ov)
        return (
            q_numeric,
            q_attn.to(torch.float32),
            q_token_ids,
            q_hidden,
            q_mask,
            a_numeric,
            a_attn.to(torch.float32),
            a_token_ids,
            a_hidden,
            a_mask,
            torch.tensor([label], dtype=torch.float32),
            idx,
        )



def _hidden_segment(sidecar: dict, name: str, max_len: int) -> torch.Tensor:
    hidden = (sidecar.get("last_hidden_states") or {}).get(name)
    if hidden is None or not hasattr(hidden, "numel") or hidden.numel() == 0:
        raise ValueError(
            f"missing last_hidden_states[{name!r}] in sidecar; re-run feature extraction with unified_answer_span_v2"
        )
    hidden = hidden.to(torch.float16)
    if hidden.dim() != 2:
        raise ValueError(f"last_hidden_states[{name!r}] must have shape [tokens, hidden_dim], got {tuple(hidden.shape)}")
    length = int(hidden.shape[0])
    if length > max_len:
        hidden = hidden[:max_len]
    elif length < max_len:
        hidden = F.pad(hidden, (0, 0, 0, max_len - length))
    return hidden.contiguous()

def _pad_attn(raw: torch.Tensor, max_len: int) -> torch.Tensor:
    if raw.shape[-1] < max_len:
        return F.pad(raw, (0, max_len - raw.shape[-1]))
    return raw[:, :max_len]


class ContextQuestionAnswerSidecarDataset(Dataset):
    """Dataset for with-context sidecar features.

    It builds three token-centered tensors over context, question, and answer.
    Each branch receives numeric token features, frozen token embeddings, and two
    layer/head attention-entropy groups:
      - context: entropy over answer->context and question->context sources
      - question: entropy over answer->question and question->context targets
      - answer: entropy over answer->question and answer->context targets
    """

    def __init__(
        self,
        meta_data,
        tokenizer,
        context_max_len=2048,
        question_max_len=128,
        answer_max_len=32,
        positive_transition="NY",
        label_mode="transition_positive",
        rag_class_labels_by_id=None,
        regression_transition="YN",
        use_numeric_features=True,
        use_probability_features=True,
        use_residual_features=True,
        use_attention_features=True,
        use_last_hidden=True,
        use_attention_mass=False,
        ablate_segments=(),
        ablate_channels=(),
        label_override=None,
        rewritten_label_override=None,
        label_override_required=False,
    ):
        if label_override_required and (label_override is not None or rewritten_label_override is not None):
            def _has_aligned_label(m):
                src = rewritten_label_override if "rewritten" in str(_get_tensor_path(m) or "") else label_override
                return src is not None and _get_example_id(m) in src
            meta_data = [m for m in meta_data if _has_aligned_label(m)]
        self.meta_data = meta_data
        self.tokenizer = tokenizer
        self.context_max_len = int(context_max_len)
        self.question_max_len = int(question_max_len)
        self.answer_max_len = int(answer_max_len)
        self.positive_transition = positive_transition
        self.label_mode = label_mode
        self.rag_class_labels_by_id = rag_class_labels_by_id or {}
        self.regression_transition = regression_transition
        self.use_numeric_features = bool(use_numeric_features)
        self.use_probability_features = bool(use_probability_features)
        self.use_residual_features = bool(use_residual_features)
        self.use_attention_features = bool(use_attention_features)
        self.use_last_hidden = bool(use_last_hidden)
        self.use_attention_mass = bool(use_attention_mass)
        # Ablation hooks (same convention as the use_* flags: content zeroed, masks
        # and dims kept so encoders/caches stay shape-compatible).
        #   ablate_segments: subset of {"context","question","answer"} -> zero that
        #     segment's numeric/hidden/attention features.
        #   ablate_channels: numeric-channel NAME SUFFIXES to zero (e.g. "_lens",
        #     "_norm" to split the residual stack).
        self.ablate_segments = frozenset(ablate_segments or ())
        self.ablate_channels = tuple(ablate_channels or ())
        # aligned-label overrides {example_id -> 0.0/1.0}. Rewritten-context
        # records (same eid as their base example) use rewritten_label_override,
        # detected via the record's tensor path.
        self.label_override = label_override
        self.rewritten_label_override = rewritten_label_override
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        prefixes = []
        if self.use_probability_features:
            prefixes.extend(["p", "entropy", "surprisal", "top1_prob", "margin"])
        if self.use_residual_features:
            prefixes.append("resid_")
        self.numeric_prefixes = tuple(prefixes)
        # packed fast path (PROBE_PACKED_ROOT): all-or-nothing per dataset, see packed_store
        self._packed = packed_store.resolve(self.meta_data) if packed_store.enabled() else None

    def __len__(self):
        return len(self.meta_data)

    def __getitems__(self, indices):
        # batch-direct packed mode: opted in per instance by the trainer's
        # fast_collate sites (fast_collate passes the ready batch through)
        if self._packed is not None and getattr(self, "_packed_batch", False):
            return packed_store.build_context_batch(self, indices)
        return _parallel_getitems(self, indices)

    @staticmethod
    def _valid_mask(span: dict, max_len: int) -> torch.Tensor:
        length = min(int(span.get("len", 0) or 0), int(max_len))
        mask = torch.zeros(int(max_len), dtype=torch.bool)
        mask[:length] = True
        return mask

    def _attn_groups(self, sidecar, c_len: int, q_len: int, a_len: int):
        maps = (sidecar.get("attention_maps") or {})
        aq = maps.get("answer_to_question")
        ac = maps.get("answer_to_context")
        qc = maps.get("question_to_context")

        layer_head = 0
        for value in (aq, ac, qc):
            if value is not None and value.numel() > 0:
                layer_head = int(value.shape[0] * value.shape[1])
                break

        def zeros(max_len):
            return torch.zeros((layer_head, int(max_len)), dtype=torch.float32)

        if layer_head == 0:
            return zeros(self.context_max_len), zeros(self.question_max_len), zeros(self.answer_max_len)

        if ac is not None and ac.numel() > 0 and c_len > 0 and a_len > 0:
            ac = ac.float()[:, :, :a_len, :c_len]
            c_from_a = _attention_entropy(ac, dim=2).reshape(-1, c_len)
            a_to_c = _attention_entropy(ac, dim=3).reshape(-1, a_len)
            c_from_a_mass = ac.sum(dim=2).reshape(-1, c_len)
            a_to_c_mass = ac.sum(dim=3).reshape(-1, a_len)
        else:
            c_from_a = torch.zeros((layer_head, 0), dtype=torch.float32)
            a_to_c = torch.zeros((layer_head, 0), dtype=torch.float32)
            c_from_a_mass = torch.zeros((layer_head, 0), dtype=torch.float32)
            a_to_c_mass = torch.zeros((layer_head, 0), dtype=torch.float32)

        if qc is not None and qc.numel() > 0 and c_len > 0 and q_len > 0:
            qc = qc.float()[:, :, :q_len, :c_len]
            c_from_q = _attention_entropy(qc, dim=2).reshape(-1, c_len)
            q_to_c = _attention_entropy(qc, dim=3).reshape(-1, q_len)
            c_from_q_mass = qc.sum(dim=2).reshape(-1, c_len)
            q_to_c_mass = qc.sum(dim=3).reshape(-1, q_len)
        else:
            c_from_q = torch.zeros((layer_head, 0), dtype=torch.float32)
            q_to_c = torch.zeros((layer_head, 0), dtype=torch.float32)
            c_from_q_mass = torch.zeros((layer_head, 0), dtype=torch.float32)
            q_to_c_mass = torch.zeros((layer_head, 0), dtype=torch.float32)

        if aq is not None and aq.numel() > 0 and q_len > 0 and a_len > 0:
            aq = aq.float()[:, :, :a_len, :q_len]
            q_from_a = _attention_entropy(aq, dim=2).reshape(-1, q_len)
            a_to_q = _attention_entropy(aq, dim=3).reshape(-1, a_len)
            q_from_a_mass = aq.sum(dim=2).reshape(-1, q_len)
            a_to_q_mass = aq.sum(dim=3).reshape(-1, a_len)
        else:
            q_from_a = torch.zeros((layer_head, 0), dtype=torch.float32)
            a_to_q = torch.zeros((layer_head, 0), dtype=torch.float32)
            q_from_a_mass = torch.zeros((layer_head, 0), dtype=torch.float32)
            a_to_q_mass = torch.zeros((layer_head, 0), dtype=torch.float32)

        # entropy groups first, then (optionally) the normalized segment-mass simplex
        # groups -- mirrors the question-only dataset's cat([ent, mass]) layout.
        c_groups = [c_from_a, c_from_q]
        q_groups = [q_from_a, q_to_c]
        a_groups = [a_to_q, a_to_c]
        if self.use_attention_mass:
            c_groups += [c_from_a_mass, c_from_q_mass]
            q_groups += [q_from_a_mass, q_to_c_mass]
            a_groups += [a_to_q_mass, a_to_c_mass]
        c_attn = torch.cat([_pad_attn(g, self.context_max_len) for g in c_groups], dim=0)
        q_attn = torch.cat([_pad_attn(g, self.question_max_len) for g in q_groups], dim=0)
        a_attn = torch.cat([_pad_attn(g, self.answer_max_len) for g in a_groups], dim=0)
        return c_attn, q_attn, a_attn

    def __getitem__(self, idx):
        item = self.meta_data[idx]
        if self._packed is not None:
            pack, row = self._packed[idx]
            return packed_store.build_context_item(self, item, pack, row, idx)
        tensor_path = _get_tensor_path(item)
        token_ids_path = _get_token_ids_path(item)
        sidecar_path = _get_sidecar_path(item)
        if not tensor_path or not os.path.exists(tensor_path):
            raise FileNotFoundError(f"missing tensor_path for idx={idx}: {tensor_path}")
        if not token_ids_path or not os.path.exists(token_ids_path):
            raise FileNotFoundError(f"missing token_ids_path for idx={idx}: {token_ids_path}")
        if not sidecar_path or not os.path.exists(sidecar_path):
            raise FileNotFoundError(f"missing sidecar_path for idx={idx}: {sidecar_path}")

        features = torch.load(tensor_path, map_location="cpu", weights_only=True).to(torch.float32)
        token_ids = torch.load(token_ids_path, map_location="cpu", weights_only=True).to(torch.long)
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)

        c_span = _packed_span(item, "context")
        q_span = _packed_span(item, "question")
        a_span = _packed_span(item, "answer")
        numeric_idx = _feature_channel_indices(item, self.numeric_prefixes)
        if not numeric_idx:
            numeric_idx = list(range(min(5, features.shape[0]))) if self.use_probability_features else [0]
        numeric = features[numeric_idx]
        if self.ablate_channels:
            names = _get_feature_names(item)
            rows = [r for r, ci in enumerate(numeric_idx)
                    if ci < len(names) and any(str(names[ci]).endswith(suf) for suf in self.ablate_channels)]
            if rows:
                numeric = numeric.clone()
                numeric[rows] = 0.0

        c_numeric = _segment_slice(numeric, c_span, self.context_max_len)
        q_numeric = _segment_slice(numeric, q_span, self.question_max_len)
        a_numeric = _segment_slice(numeric, a_span, self.answer_max_len)
        if not self.use_numeric_features:
            c_numeric = torch.zeros_like(c_numeric)
            q_numeric = torch.zeros_like(q_numeric)
            a_numeric = torch.zeros_like(a_numeric)
        c_token_ids = _segment_slice(token_ids, c_span, self.context_max_len, fill_value=self.pad_id).to(torch.long)
        q_token_ids = _segment_slice(token_ids, q_span, self.question_max_len, fill_value=self.pad_id).to(torch.long)
        a_token_ids = _segment_slice(token_ids, a_span, self.answer_max_len, fill_value=self.pad_id).to(torch.long)
        c_hidden = _hidden_segment(sidecar, "context", self.context_max_len)
        q_hidden = _hidden_segment(sidecar, "question", self.question_max_len)
        a_hidden = _hidden_segment(sidecar, "answer", self.answer_max_len)
        if not self.use_last_hidden:
            c_hidden = torch.zeros_like(c_hidden)
            q_hidden = torch.zeros_like(q_hidden)
            a_hidden = torch.zeros_like(a_hidden)

        c_len = min(int(c_span.get("len", 0) or 0), self.context_max_len)
        q_len = min(int(q_span.get("len", 0) or 0), self.question_max_len)
        a_len = min(int(a_span.get("len", 0) or 0), self.answer_max_len)
        c_mask = self._valid_mask(c_span, self.context_max_len)
        q_mask = self._valid_mask(q_span, self.question_max_len)
        a_mask = self._valid_mask(a_span, self.answer_max_len)
        c_attn, q_attn, a_attn = self._attn_groups(sidecar, c_len=c_len, q_len=q_len, a_len=a_len)
        if not self.use_attention_features:
            c_attn = torch.zeros_like(c_attn)
            q_attn = torch.zeros_like(q_attn)
            a_attn = torch.zeros_like(a_attn)
        if "context" in self.ablate_segments:
            c_numeric = torch.zeros_like(c_numeric)
            c_hidden = torch.zeros_like(c_hidden)
            c_attn = torch.zeros_like(c_attn)
        if "question" in self.ablate_segments:
            q_numeric = torch.zeros_like(q_numeric)
            q_hidden = torch.zeros_like(q_hidden)
            q_attn = torch.zeros_like(q_attn)
        if "answer" in self.ablate_segments:
            a_numeric = torch.zeros_like(a_numeric)
            a_hidden = torch.zeros_like(a_hidden)
            a_attn = torch.zeros_like(a_attn)

        label = get_binary_label(item, self.positive_transition, self.label_mode)
        if self.label_override is not None or self.rewritten_label_override is not None:
            is_rew = "rewritten" in str(_get_tensor_path(item) or "")
            src = self.rewritten_label_override if is_rew else self.label_override
            ov = src.get(_get_example_id(item)) if src is not None else None
            if ov is not None:
                label = float(ov)
        aux_binary_labels = get_aux_binary_labels(item)
        rag_class_label = self.rag_class_labels_by_id.get(
            _get_example_id(item),
            get_rag_signal_label(item, regression_transition=self.regression_transition),
        )
        return (
            c_numeric,
            c_attn.to(torch.float32),
            c_token_ids,
            c_hidden,
            c_mask,
            q_numeric,
            q_attn.to(torch.float32),
            q_token_ids,
            q_hidden,
            q_mask,
            a_numeric,
            a_attn.to(torch.float32),
            a_token_ids,
            a_hidden,
            a_mask,
            torch.tensor([label], dtype=torch.float32),
            torch.tensor(rag_class_label, dtype=torch.long),
            torch.tensor(aux_binary_labels, dtype=torch.float32),
            idx,
        )




def _pair_key(item: dict) -> tuple[str, str]:
    ex = _get_example(item)
    return (str(ex.get("dataset") or ""), str(ex.get("id") or ""))


class PairedQuestionContextSidecarDataset(Dataset):
    """Join question-only and with-context sidecar records by dataset/example id."""

    def __init__(
        self,
        question_meta,
        context_meta,
        tokenizer,
        question_max_len=128,
        answer_max_len=32,
        context_max_len=2048,
        use_attention_mass=False,
    ):
        q_by_key = {_pair_key(item): item for item in question_meta}
        self.pairs = []
        for c_item in context_meta:
            q_item = q_by_key.get(_pair_key(c_item))
            if q_item is not None:
                self.pairs.append((q_item, c_item))
        self.question_ds = QuestionAnswerSidecarDataset(
            [p[0] for p in self.pairs],
            tokenizer=tokenizer,
            question_max_len=question_max_len,
            answer_max_len=answer_max_len,
            label_mode="without_context_correct",
            use_attention_mass=use_attention_mass,
        )
        self.context_ds = ContextQuestionAnswerSidecarDataset(
            [p[1] for p in self.pairs],
            tokenizer=tokenizer,
            context_max_len=context_max_len,
            question_max_len=question_max_len,
            answer_max_len=answer_max_len,
            label_mode="with_context_correct",
            use_attention_mass=use_attention_mass,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qa = self.question_ds[idx]
        cqa = self.context_ds[idx]
        q_item, c_item = self.pairs[idx]
        no_label = torch.tensor([1.0 if get_without_context_correct(q_item) else 0.0], dtype=torch.float32)
        ctx_label = torch.tensor([1.0 if get_with_context_correct(c_item) else 0.0], dtype=torch.float32)
        return qa[:-2], cqa[:-4], no_label, ctx_label, idx
