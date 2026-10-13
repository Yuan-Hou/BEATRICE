"""Runtime reader for packed sidecar-derived features.

Pack format: scripts/pack_sidecar_features.py. Activation: set PROBE_PACKED_ROOT
(e.g. data/packed_features). The sidecar datasets then resolve every meta row
against the packed index at construction time; if — and only if — every row is
covered, __getitem__ switches to the packed fast path (mmap slice + zero-pad).
Any uncovered row keeps the whole dataset on the slow sidecar path, so results
can never silently mix.

Bit-exactness: the builders below mirror QuestionAnswerSidecarDataset.__getitem__
and ContextQuestionAnswerSidecarDataset.__getitem__ op-for-op on the packed values
(which are stored exactly as the slow path computes them, trimmed). Verified by
scripts/verify_packed_features.py with torch.equal over full output tuples.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_DTYPES = {"feats": np.float16, "tok": np.int32, "hid": np.float16, "attn": np.float32}
_LOCK = threading.Lock()
_INDEX = None          # {tensor_path: (Pack, row)}
_WARNED = set()


class Pack:
    def __init__(self, d: Path):
        ix = torch.load(d / "index.pt", weights_only=False)
        self.dir = d
        self.lens = ix["lens"]
        self.offsets = ix["offsets"]
        self.chan = ix["chan"]
        self.segs = ix["segs"]
        self.tensor_paths = ix["tensor_path"]
        self._mm = {}

    def _memmap(self, field: str):
        m = self._mm.get(field)
        if m is None:
            dt = _DTYPES[field.split("_")[0]]
            path = self.dir / f"{field}.{np.dtype(dt).name}.bin"
            self._mm[field] = m = np.memmap(path, dtype=dt, mode="r")
        return m

    def read(self, field: str, row: int, span_len: int) -> torch.Tensor:
        off = self.offsets[field]
        flat = self._memmap(field)[off[row]:off[row + 1]]
        kind = field.split("_")[0]
        if kind == "tok":
            return torch.from_numpy(np.ascontiguousarray(flat))
        c = self.chan.get(field) or 0
        if kind == "hid":
            return torch.from_numpy(np.ascontiguousarray(flat).reshape(-1, c))
        if c == 0:  # zero-channel attention (no maps in this file)
            return torch.zeros((0, span_len), dtype=torch.float32)
        return torch.from_numpy(np.ascontiguousarray(flat).reshape(c, -1))


def enabled() -> bool:
    root = os.environ.get("PROBE_PACKED_ROOT")
    return bool(root) and Path(root).is_dir()


def _index():
    global _INDEX
    with _LOCK:
        if _INDEX is None:
            idx = {}
            root = Path(os.environ["PROBE_PACKED_ROOT"])
            for ixp in sorted(root.glob("*/*/index.pt")):
                p = Pack(ixp.parent)
                for i, tp in enumerate(p.tensor_paths):
                    idx[tp] = (p, i)
            _INDEX = idx
        return _INDEX


def resolve(meta_data) -> list | None:
    """Map every meta row to (Pack, row); None (-> slow path) if any row is missing."""
    from . import dataset as _ds
    idx = _index()
    out, missing = [], 0
    for item in meta_data:
        hit = idx.get(_ds._get_tensor_path(item))
        if hit is None:
            missing += 1
        out.append(hit)
    if missing:
        key = f"missing={missing}/{len(meta_data)}"
        if key not in _WARNED:
            _WARNED.add(key)
            print(f"[packed_store] {missing}/{len(meta_data)} rows not packed -> slow sidecar path", flush=True)
        return None
    return out


def _pad_tail(x: torch.Tensor, cap: int, value=0) -> torch.Tensor:
    l = int(x.shape[-1])
    if l > cap:
        return x[..., :cap]
    if l < cap:
        return F.pad(x, (0, cap - l), value=value)
    return x


def _pad_rows(x: torch.Tensor, cap: int) -> torch.Tensor:
    l = int(x.shape[0])
    if l > cap:
        return x[:cap]
    if l < cap:
        return F.pad(x, (0, 0, 0, cap - l))
    return x


def _numeric(ds, item, feats_f16: torch.Tensor, cap: int, ablate_channels=()) -> torch.Tensor:
    from . import dataset as _ds
    features = feats_f16.to(torch.float32)
    numeric_idx = _ds._feature_channel_indices(item, ds.numeric_prefixes)
    if not numeric_idx:
        numeric_idx = list(range(min(5, features.shape[0]))) if ds.use_probability_features else [0]
    numeric = features[numeric_idx]
    if ablate_channels:
        names = _ds._get_feature_names(item)
        rows = [r for r, ci in enumerate(numeric_idx)
                if ci < len(names) and any(str(names[ci]).endswith(suf) for suf in ablate_channels)]
        if rows:
            numeric = numeric.clone()
            numeric[rows] = 0.0
    return _pad_tail(numeric, cap)


def _attn(ds, raw: torch.Tensor, cap: int) -> torch.Tensor:
    """raw holds the full ent+mass group stack; without use_attention_mass only the
    leading entropy half is used (group order matches the slow path's cat)."""
    if raw.shape[0] and not ds.use_attention_mass:
        raw = raw[: raw.shape[0] // 2]
    return _pad_tail(raw.to(torch.float32), cap)


def _numeric_idx(ds, item):
    from . import dataset as _ds
    ni = _ds._feature_channel_indices(item, ds.numeric_prefixes)
    if not ni:
        # mirrors the slow path's fallback (feature stacks always have >5 channels)
        ni = list(range(5)) if ds.use_probability_features else [0]
    return ni


def _batch_common(ds, indices):
    from concurrent.futures import ThreadPoolExecutor
    packs = [ds._packed[i] for i in indices]
    items = [ds.meta_data[i] for i in indices]
    return packs, items, ThreadPoolExecutor(max_workers=min(16, max(1, len(indices))))


class _BatchBufs:
    """Reused batch buffers with per-slot dirty-length tracking.

    Re-zeroing a fresh 5.6GB ctx batch every fetch costs ~1s of memset; instead we
    keep the buffers alive on the dataset instance and, per slot, clear only the
    region the previous batch actually wrote (~12% on average). Safe because the
    encoder loaders run num_workers=0 / pin_memory=False: the consumer finishes
    with a batch (copies it to GPU) before the next fetch overwrites it.

    With `pin=True` (PROBE_CUDA_PREFETCH) buffers are allocated in pinned memory
    and rotate over `nslots=2` sets, so CudaPrefetchLoader's background thread can
    fill batch N+1 while batch N's async H2D copy is still in flight; reuse of a
    set is fenced by that copy's CUDA event (see CudaPrefetchLoader.worker).
    """

    def __init__(self, nslots: int = 1, pin: bool = False):
        self.nslots = int(nslots)
        self.pin = bool(pin)
        self.slot = 0
        self.bufs: dict = {}
        self.dirty: dict = {}

    def begin_batch(self):
        self.slot = (self.slot + 1) % self.nslots

    def _k(self, key):
        return (self.slot, key)

    def get(self, key, B, tail_shape, dtype, fill=0):
        k = self._k(key)
        buf = self.bufs.get(k)
        if buf is None or buf.shape[0] < B or tuple(buf.shape[1:]) != tuple(tail_shape):
            buf = (torch.full((B, *tail_shape), fill, dtype=dtype, pin_memory=self.pin) if fill
                   else torch.zeros((B, *tail_shape), dtype=dtype, pin_memory=self.pin))
            self.bufs[k] = buf
            self.dirty[k] = [0] * B
        return buf[:B]

    def clear_slot(self, key, i, fill=0):
        k = self._k(key)
        prev = self.dirty[k][i]
        if prev:
            buf = self.bufs[k]
            if buf.dim() == 2:            # [B, L] (tok / mask)
                buf[i, :prev] = fill
            elif key.endswith("_hid"):    # [B, L, D]: length is dim 1
                buf[i, :prev] = 0
            else:                          # [B, C, L]
                buf[i, :, :prev] = 0
        self.dirty[k][i] = 0

    def mark(self, key, i, length):
        k = self._k(key)
        if length > self.dirty[k][i]:
            self.dirty[k][i] = length


class CudaPrefetchLoader:
    """Overlap packed CPU fetch and H2D copies with GPU compute (train loaders only).

    A background thread drives the wrapped DataLoader (whose dataset fills rotating
    pinned _BatchBufs), enqueues non-blocking H2D copies on a dedicated stream and
    hands over ready GPU batches through a depth-1 queue. The consumer's per-field
    `.to(device)` calls become no-ops. Values are bit-identical to the synchronous
    path: only the copy scheduling changes.

    Safety fences: buffer set s is refilled only after the CUDA event of the
    previous batch that used s has completed (worker-side synchronize); the main
    stream waits on each batch's event before compute and record_stream()s the
    tensors. Requires num_workers=0 and DataLoader pin_memory=False.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = torch.device(device)
        self._evs = [None, None]

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        import queue as _q
        for ev in self._evs:      # epoch boundary: quiesce previous run's copies
            if ev is not None:
                ev.synchronize()
        self._evs = [None, None]
        q: _q.Queue = _q.Queue(maxsize=1)
        stream = torch.cuda.Stream(self.device)
        SENT = object()
        err: list = []

        def worker():
            try:
                i = 0
                it = iter(self.loader)
                while True:
                    prev = self._evs[i % 2]
                    if prev is not None:  # fence: slot about to be refilled
                        prev.synchronize()
                    try:
                        batch = next(it)
                    except StopIteration:
                        break
                    with torch.cuda.stream(stream):
                        gpu = tuple(x.to(self.device, non_blocking=True)
                                    if isinstance(x, torch.Tensor) else x for x in batch)
                    ev = torch.cuda.Event()
                    ev.record(stream)
                    self._evs[i % 2] = ev
                    q.put((gpu, ev))
                    i += 1
            except BaseException as e:  # surfaced on the consumer side
                err.append(e)
            finally:
                q.put(SENT)

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        while True:
            item = q.get()
            if item is SENT:
                if err:
                    raise err[0]
                return
            gpu, ev = item
            cur = torch.cuda.current_stream(self.device)
            cur.wait_event(ev)
            for x in gpu:
                if isinstance(x, torch.Tensor):
                    x.record_stream(cur)
            yield gpu


def maybe_cuda_prefetch(loader, ds, device):
    """Wrap a train DataLoader with CudaPrefetchLoader when the packed batch-direct
    path is active and PROBE_CUDA_PREFETCH=1. Must be called before first iteration
    (switches the dataset's buffers to pinned dual-slot mode)."""
    if (os.environ.get("PROBE_CUDA_PREFETCH") == "1"
            and getattr(ds, "_packed", None) is not None
            and getattr(ds, "_packed_batch", False)
            and str(device).startswith("cuda") and torch.cuda.is_available()):
        ds._packed_pin = True
        ds._batch_bufs = None   # drop any unpinned single-slot buffers
        print("[packed_store] CUDA prefetch enabled (pinned dual-slot buffers)", flush=True)
        return CudaPrefetchLoader(loader, device)
    return loader


def build_question_batch(ds, indices):
    """Batch-direct packed fetch: write only valid bytes into zeroed batch buffers.

    Output is element-wise torch.equal to fast_collate([__getitem__(i) ...]) —
    padding regions are zeros (pad_id for token ids) exactly as the per-item path
    pads them, and flag/ablation zeroing becomes skipped writes.
    """
    from . import dataset as _ds
    B = len(indices)
    packs, items, pool = _batch_common(ds, indices)
    qc, ac = ds.question_max_len, ds.answer_max_len
    ni = _numeric_idx(ds, items[0])
    a_chan = packs[0][0].chan.get("attn_q") or 0
    if a_chan and not ds.use_attention_mass:
        a_chan //= 2
    D = packs[0][0].chan.get("hid_q")
    bufs = getattr(ds, "_batch_bufs", None)
    if bufs is None:
        pin = bool(getattr(ds, "_packed_pin", False))
        bufs = ds._batch_bufs = _BatchBufs(nslots=2 if pin else 1, pin=pin)
    bufs.begin_batch()
    caps = {"q": qc, "a": ac}
    out = {}
    for s, cap in caps.items():
        out[f"{s}_num"] = bufs.get(f"{s}_num", B, (len(ni), cap), torch.float32)
        out[f"{s}_attn"] = bufs.get(f"{s}_attn", B, (a_chan, cap), torch.float32)
        out[f"{s}_tok"] = bufs.get(f"{s}_tok", B, (cap,), torch.long, fill=ds.pad_id)
        out[f"{s}_hid"] = bufs.get(f"{s}_hid", B, (cap, D), torch.float16)
        out[f"{s}_mask"] = bufs.get(f"{s}_mask", B, (cap,), torch.bool)
    labels = torch.empty((B, 1), dtype=torch.float32)

    def fill(i):
        pack, row = packs[i]
        item = items[i]
        nidx = _numeric_idx(ds, item)
        for s, cap in caps.items():
            sl = min(pack.lens[s]["span"][row], cap)
            lf = min(pack.lens[s]["feat"][row], cap)
            lh = min(pack.lens[s]["hid"][row], cap)
            seg_ok = {"q": "question", "a": "answer"}[s] not in ds.ablate_segments
            for suf, fill_v in (("_num", 0), ("_attn", 0), ("_hid", 0), ("_mask", False), ("_tok", ds.pad_id)):
                bufs.clear_slot(f"{s}{suf}", i, fill_v)
            if lf:
                out[f"{s}_tok"][i, :lf] = pack.read(f"tok_{s}", row, sl)[:lf].to(torch.long)
                bufs.mark(f"{s}_tok", i, lf)
                if ds.use_numeric_features and seg_ok:
                    feats = pack.read(f"feats_{s}", row, sl).to(torch.float32)[nidx]
                    out[f"{s}_num"][i, :, :lf] = feats[:, :lf]
                    bufs.mark(f"{s}_num", i, lf)
            if lh and ds.use_last_hidden and seg_ok:
                out[f"{s}_hid"][i, :lh] = pack.read(f"hid_{s}", row, sl)[:lh]
                bufs.mark(f"{s}_hid", i, lh)
            out[f"{s}_mask"][i, :sl] = True
            bufs.mark(f"{s}_mask", i, sl)
            if a_chan and ds.use_attention_features and seg_ok:
                raw = pack.read(f"attn_{s}", row, sl)
                out[f"{s}_attn"][i, :, :raw.shape[1]] = raw[:a_chan]
                bufs.mark(f"{s}_attn", i, raw.shape[1])
        label = _ds.get_binary_label(item, ds.positive_transition, ds.label_mode)
        if ds.label_override is not None:
            ov = ds.label_override.get(_ds._get_example_id(item))
            if ov is not None:
                label = float(ov)
        labels[i, 0] = label

    with pool as ex:
        list(ex.map(fill, range(B)))
    return (out["q_num"], out["q_attn"], out["q_tok"], out["q_hid"], out["q_mask"],
            out["a_num"], out["a_attn"], out["a_tok"], out["a_hid"], out["a_mask"],
            labels, torch.as_tensor(list(indices), dtype=torch.int64))


def build_context_batch(ds, indices):
    """With-context analogue of build_question_batch (see its contract)."""
    from . import dataset as _ds
    B = len(indices)
    packs, items, pool = _batch_common(ds, indices)
    cc, qc, ac = ds.context_max_len, ds.question_max_len, ds.answer_max_len
    ni = _numeric_idx(ds, items[0])
    a_chan = packs[0][0].chan.get("attn_c") or 0
    if a_chan and not ds.use_attention_mass:
        a_chan //= 2
    D = packs[0][0].chan.get("hid_c")
    bufs = getattr(ds, "_batch_bufs", None)
    if bufs is None:
        pin = bool(getattr(ds, "_packed_pin", False))
        bufs = ds._batch_bufs = _BatchBufs(nslots=2 if pin else 1, pin=pin)
    bufs.begin_batch()
    caps = {"c": cc, "q": qc, "a": ac}
    out = {}
    for s, cap in caps.items():
        out[f"{s}_num"] = bufs.get(f"{s}_num", B, (len(ni), cap), torch.float32)
        out[f"{s}_attn"] = bufs.get(f"{s}_attn", B, (a_chan, cap), torch.float32)
        out[f"{s}_tok"] = bufs.get(f"{s}_tok", B, (cap,), torch.long, fill=ds.pad_id)
        out[f"{s}_hid"] = bufs.get(f"{s}_hid", B, (cap, D), torch.float16)
        out[f"{s}_mask"] = bufs.get(f"{s}_mask", B, (cap,), torch.bool)
    labels = torch.empty((B, 1), dtype=torch.float32)
    rag_cls = torch.empty((B,), dtype=torch.long)
    aux_list: list = [None] * B

    def fill(i):
        pack, row = packs[i]
        item = items[i]
        nidx = _numeric_idx(ds, item)
        ab_rows = []
        if ds.ablate_channels:
            names = _ds._get_feature_names(item)
            ab_rows = [r for r, ci in enumerate(nidx)
                       if ci < len(names) and any(str(names[ci]).endswith(suf) for suf in ds.ablate_channels)]
        for s, cap in caps.items():
            sl = min(pack.lens[s]["span"][row], cap)
            lf = min(pack.lens[s]["feat"][row], cap)
            lh = min(pack.lens[s]["hid"][row], cap)
            seg_ok = {"c": "context", "q": "question", "a": "answer"}[s] not in ds.ablate_segments
            for suf, fill_v in (("_num", 0), ("_attn", 0), ("_hid", 0), ("_mask", False), ("_tok", ds.pad_id)):
                bufs.clear_slot(f"{s}{suf}", i, fill_v)
            if lf:
                out[f"{s}_tok"][i, :lf] = pack.read(f"tok_{s}", row, sl)[:lf].to(torch.long)
                bufs.mark(f"{s}_tok", i, lf)
                if ds.use_numeric_features and seg_ok:
                    feats = pack.read(f"feats_{s}", row, sl).to(torch.float32)[nidx]
                    out[f"{s}_num"][i, :, :lf] = feats[:, :lf]
                    bufs.mark(f"{s}_num", i, lf)
                    if ab_rows:
                        out[f"{s}_num"][i, ab_rows, :lf] = 0.0
            if lh and ds.use_last_hidden and seg_ok:
                out[f"{s}_hid"][i, :lh] = pack.read(f"hid_{s}", row, sl)[:lh]
                bufs.mark(f"{s}_hid", i, lh)
            out[f"{s}_mask"][i, :sl] = True
            bufs.mark(f"{s}_mask", i, sl)
            if a_chan and ds.use_attention_features and seg_ok:
                raw = pack.read(f"attn_{s}", row, sl)
                out[f"{s}_attn"][i, :, :raw.shape[1]] = raw[:a_chan]
                bufs.mark(f"{s}_attn", i, raw.shape[1])
        label = _ds.get_binary_label(item, ds.positive_transition, ds.label_mode)
        if ds.label_override is not None or ds.rewritten_label_override is not None:
            is_rew = "rewritten" in str(_ds._get_tensor_path(item) or "")
            src = ds.rewritten_label_override if is_rew else ds.label_override
            ov = src.get(_ds._get_example_id(item)) if src is not None else None
            if ov is not None:
                label = float(ov)
        labels[i, 0] = label
        rag_cls[i] = ds.rag_class_labels_by_id.get(
            _ds._get_example_id(item),
            _ds.get_rag_signal_label(item, regression_transition=ds.regression_transition))
        aux_list[i] = torch.tensor(_ds.get_aux_binary_labels(item), dtype=torch.float32)

    with pool as ex:
        list(ex.map(fill, range(B)))
    return (out["c_num"], out["c_attn"], out["c_tok"], out["c_hid"], out["c_mask"],
            out["q_num"], out["q_attn"], out["q_tok"], out["q_hid"], out["q_mask"],
            out["a_num"], out["a_attn"], out["a_tok"], out["a_hid"], out["a_mask"],
            labels, rag_cls, torch.stack(aux_list),
            torch.as_tensor(list(indices), dtype=torch.int64))


def build_question_item(ds, item, pack: Pack, row: int, idx: int):
    from . import dataset as _ds
    L = pack.lens
    q_slen, a_slen = L["q"]["span"][row], L["a"]["span"][row]

    q_numeric = _numeric(ds, item, pack.read("feats_q", row, q_slen), ds.question_max_len)
    a_numeric = _numeric(ds, item, pack.read("feats_a", row, a_slen), ds.answer_max_len)
    if not ds.use_numeric_features:
        q_numeric = torch.zeros_like(q_numeric)
        a_numeric = torch.zeros_like(a_numeric)
    q_token_ids = _pad_tail(pack.read("tok_q", row, q_slen).to(torch.long), ds.question_max_len, value=ds.pad_id)
    a_token_ids = _pad_tail(pack.read("tok_a", row, a_slen).to(torch.long), ds.answer_max_len, value=ds.pad_id)
    q_hidden = _pad_rows(pack.read("hid_q", row, q_slen), ds.question_max_len)
    a_hidden = _pad_rows(pack.read("hid_a", row, a_slen), ds.answer_max_len)
    if not ds.use_last_hidden:
        q_hidden = torch.zeros_like(q_hidden)
        a_hidden = torch.zeros_like(a_hidden)

    q_len = min(q_slen, ds.question_max_len)
    a_len = min(a_slen, ds.answer_max_len)
    q_mask = torch.zeros(ds.question_max_len, dtype=torch.bool)
    a_mask = torch.zeros(ds.answer_max_len, dtype=torch.bool)
    q_mask[:q_len] = True
    a_mask[:a_len] = True

    q_attn = _attn(ds, pack.read("attn_q", row, q_slen), ds.question_max_len)
    a_attn = _attn(ds, pack.read("attn_a", row, a_slen), ds.answer_max_len)
    if not ds.use_attention_features:
        q_attn = torch.zeros_like(q_attn)
        a_attn = torch.zeros_like(a_attn)
    if "question" in ds.ablate_segments:
        q_numeric, q_hidden, q_attn = torch.zeros_like(q_numeric), torch.zeros_like(q_hidden), torch.zeros_like(q_attn)
    if "answer" in ds.ablate_segments:
        a_numeric, a_hidden, a_attn = torch.zeros_like(a_numeric), torch.zeros_like(a_hidden), torch.zeros_like(a_attn)

    label = _ds.get_binary_label(item, ds.positive_transition, ds.label_mode)
    if ds.label_override is not None:
        ov = ds.label_override.get(_ds._get_example_id(item))
        if ov is not None:
            label = float(ov)
    return (q_numeric, q_attn, q_token_ids, q_hidden, q_mask,
            a_numeric, a_attn, a_token_ids, a_hidden, a_mask,
            torch.tensor([label], dtype=torch.float32), idx)


def build_context_item(ds, item, pack: Pack, row: int, idx: int):
    from . import dataset as _ds
    L = pack.lens
    c_slen, q_slen, a_slen = L["c"]["span"][row], L["q"]["span"][row], L["a"]["span"][row]

    c_numeric = _numeric(ds, item, pack.read("feats_c", row, c_slen), ds.context_max_len, ds.ablate_channels)
    q_numeric = _numeric(ds, item, pack.read("feats_q", row, q_slen), ds.question_max_len, ds.ablate_channels)
    a_numeric = _numeric(ds, item, pack.read("feats_a", row, a_slen), ds.answer_max_len, ds.ablate_channels)
    if not ds.use_numeric_features:
        c_numeric = torch.zeros_like(c_numeric)
        q_numeric = torch.zeros_like(q_numeric)
        a_numeric = torch.zeros_like(a_numeric)
    c_token_ids = _pad_tail(pack.read("tok_c", row, c_slen).to(torch.long), ds.context_max_len, value=ds.pad_id)
    q_token_ids = _pad_tail(pack.read("tok_q", row, q_slen).to(torch.long), ds.question_max_len, value=ds.pad_id)
    a_token_ids = _pad_tail(pack.read("tok_a", row, a_slen).to(torch.long), ds.answer_max_len, value=ds.pad_id)
    c_hidden = _pad_rows(pack.read("hid_c", row, c_slen), ds.context_max_len)
    q_hidden = _pad_rows(pack.read("hid_q", row, q_slen), ds.question_max_len)
    a_hidden = _pad_rows(pack.read("hid_a", row, a_slen), ds.answer_max_len)
    if not ds.use_last_hidden:
        c_hidden = torch.zeros_like(c_hidden)
        q_hidden = torch.zeros_like(q_hidden)
        a_hidden = torch.zeros_like(a_hidden)

    c_len = min(c_slen, ds.context_max_len)
    q_len = min(q_slen, ds.question_max_len)
    a_len = min(a_slen, ds.answer_max_len)
    c_mask = ds._valid_mask({"len": c_slen}, ds.context_max_len)
    q_mask = ds._valid_mask({"len": q_slen}, ds.question_max_len)
    a_mask = ds._valid_mask({"len": a_slen}, ds.answer_max_len)

    c_attn = _attn(ds, pack.read("attn_c", row, c_len), ds.context_max_len)
    q_attn = _attn(ds, pack.read("attn_q", row, q_len), ds.question_max_len)
    a_attn = _attn(ds, pack.read("attn_a", row, a_len), ds.answer_max_len)
    if not ds.use_attention_features:
        c_attn = torch.zeros_like(c_attn)
        q_attn = torch.zeros_like(q_attn)
        a_attn = torch.zeros_like(a_attn)
    if "context" in ds.ablate_segments:
        c_numeric, c_hidden, c_attn = torch.zeros_like(c_numeric), torch.zeros_like(c_hidden), torch.zeros_like(c_attn)
    if "question" in ds.ablate_segments:
        q_numeric, q_hidden, q_attn = torch.zeros_like(q_numeric), torch.zeros_like(q_hidden), torch.zeros_like(q_attn)
    if "answer" in ds.ablate_segments:
        a_numeric, a_hidden, a_attn = torch.zeros_like(a_numeric), torch.zeros_like(a_hidden), torch.zeros_like(a_attn)

    label = _ds.get_binary_label(item, ds.positive_transition, ds.label_mode)
    if ds.label_override is not None or ds.rewritten_label_override is not None:
        is_rew = "rewritten" in str(_ds._get_tensor_path(item) or "")
        src = ds.rewritten_label_override if is_rew else ds.label_override
        ov = src.get(_ds._get_example_id(item)) if src is not None else None
        if ov is not None:
            label = float(ov)
    aux_binary_labels = _ds.get_aux_binary_labels(item)
    rag_class_label = ds.rag_class_labels_by_id.get(
        _ds._get_example_id(item),
        _ds.get_rag_signal_label(item, regression_transition=ds.regression_transition),
    )
    return (c_numeric, c_attn, c_token_ids, c_hidden, c_mask,
            q_numeric, q_attn, q_token_ids, q_hidden, q_mask,
            a_numeric, a_attn, a_token_ids, a_hidden, a_mask,
            torch.tensor([label], dtype=torch.float32),
            torch.tensor(rag_class_label, dtype=torch.long),
            torch.tensor(aux_binary_labels, dtype=torch.float32),
            idx)
