"""Online probe scorer: features -> trained frozen-fusion probe -> P(answer correct).

This is the heart of the RAG probe HTTP service. Given a question (closed-book) or
context+question (RAG) and an answer (caller-provided, e.g. from vLLM — or generated
here as a convenience), it:

  1. runs the EXACT training-time feature extraction in one forward
     (`_extract_one_unified` with hidden_layer=17 + resid-lens channels inline),
  2. pushes the features through the SAME dataset transforms and frozen branch
     encoders used in training/eval (`_context/_question_dataset_from_encoder_ckpt`
     + the frozen fusion probe loaded exactly like `evaluate_probe`),
  3. returns the fusion probe's P(correct) plus per-branch probabilities.

Zero feature drift by construction: every step is the offline pipeline's own code.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ragprobe.features.extractor import _extract_one_unified
from ragprobe.training.trainer import (
    ContextQuestionAnswerFrozenFusionDataset,
    QuestionAnswerFrozenFusionDataset,
    _build_context_question_answer_frozen_fusion,
    _build_question_answer_frozen_fusion,
    _context_dataset_from_encoder_ckpt,
    _move_branch_batches,
    _question_dataset_from_encoder_ckpt,
)
from ragprobe.utils.logging import log

# training-time extraction parameters (must match the offline feature configs)
MODE_DEFAULTS = {
    "context": {"max_len": 2048, "answer_tokens": 32, "prob_top_k": 20},
    "question": {"max_len": 512, "answer_tokens": 32, "prob_top_k": 20},
}


class _QOnlyCache:
    """Bounded, thread-safe TTL+LRU store for the q-only branch vectors of a request.

    Lifecycle: /score_question puts one entry (keyed by a fresh uuid) and returns the
    uuid; /score_context reads it — possibly MANY times, once per retrieved-context
    re-scoring — so a read NEVER evicts. Entries expire only by TTL (age) or when the
    store is over capacity (least-recently-used first). Vectors are held on CPU (a few
    KB each) so the cache never pins GPU memory.
    """

    def __init__(self, max_entries: int = 20000, ttl_s: float = 1800.0):
        self.max_entries = int(max_entries)
        self.ttl_s = float(ttl_s)
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    def _expired(self, entry: dict, now: float) -> bool:
        return self.ttl_s > 0 and (now - entry["created_at"]) > self.ttl_s

    def put(self, payload: dict) -> str:
        key = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._store[key] = {**payload, "created_at": now}
            self._store.move_to_end(key)
            # evict expired + over-capacity (LRU = front)
            while self._store:
                oldest_k, oldest_v = next(iter(self._store.items()))
                if self._expired(oldest_v, now) or len(self._store) > self.max_entries:
                    self._store.popitem(last=False)
                else:
                    break
        return key

    def get(self, key: str) -> dict | None:
        if not key:
            return None
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if self._expired(entry, now):
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)  # refresh LRU on read; do NOT remove
            return entry

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._store), "max_entries": self.max_entries, "ttl_s": self.ttl_s}


class _DualHead(nn.Module):
    """Late-fusion MLP over [ctx branch vecs ‖ q-only branch vecs] -> P(with_context_correct).

    Branch order (and per-branch input dim) is taken from the checkpoint so the online
    concat is identical to training. Two checkpoint meta formats are supported:
      - joint search heads (e.g. joint_gk3): `branches` with a "q_" prefix marking the
        closed-book forwards + `branch_dims` — q-only vecs were right-zero-padded to the
        ctx vec dim (128->256) at training time, so we pad the same way here;
      - hand-built dual heads (dual_qctx): `ctx_branches`/`qonly_branches`, native dims.
    """

    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "branches" in ck:  # joint-pool format
            dims = ck.get("branch_dims") or [None] * len(ck["branches"])
            self.parts = [("q", b[2:], d) if b.startswith("q_") else ("ctx", b, d)
                          for b, d in zip(ck["branches"], dims)]
        else:  # hand-built dual format
            cd, qd = ck.get("ctx_branch_dim"), ck.get("qonly_branch_dim")
            self.parts = ([("ctx", b, cd) for b in ck["ctx_branches"]] +
                          [("q", b, qd) for b in ck["qonly_branches"]])
        self.ctx_branches = [n for s, n, _ in self.parts if s == "ctx"]
        self.qonly_branches = [n for s, n, _ in self.parts if s == "q"]
        din = int(ck["input_dim"])
        self.net = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
        self.net.load_state_dict({k.replace("net.", "", 1) if k.startswith("net.") else k: v
                                  for k, v in ck["state_dict"].items()})
        self.to(device).eval()
        self.device = device
        self.ckpt_path = ckpt_path

    def fuse(self, ctx_vecs: dict, q_vecs: dict) -> float:
        parts = []
        for source, name, dim in self.parts:
            v = (ctx_vecs if source == "ctx" else q_vecs)[name].reshape(-1)
            if dim and v.shape[0] < dim:
                v = torch.nn.functional.pad(v, (0, dim - v.shape[0]))
            parts.append(v)
        x = torch.cat(parts, dim=-1).reshape(1, -1).to(self.device)
        with torch.no_grad():
            return float(torch.sigmoid(self.net(x)).reshape(-1)[0])


def _dtype_from_string(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _normalize_contexts(raw) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"text": raw}]
    out = []
    for c in raw:
        if isinstance(c, dict):
            out.append({"text": c.get("text") or c.get("content") or c.get("paragraph_text") or ""})
        else:
            out.append({"text": str(c)})
    return out


class _FusionStack:
    """A loaded frozen-fusion probe + everything needed to feed it one record."""

    def __init__(self, mode: str, checkpoint_path: str, model_name: str, embed_weights, device: str):
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.positive_transition = ck.get("positive_transition", "NY")
        self.label_mode = ck.get("label_mode",
                                 "with_context_correct" if mode == "context" else "without_context_correct")
        self.binary_target_name = ck.get("binary_target_name", self.label_mode)
        # identical reconstruction to evaluator._evaluate_question_answer_frozen_encoder_fusion
        eval_config = {
            "model_name": model_name,
            "probe": {
                "branch_checkpoints": ck["branch_checkpoints"],
                "fusion_hidden_dim": ck.get("hidden_dim", 128),
                "dropout": ck.get("dropout", 0.1),
                "fusion_num_res_blocks": ck.get("fusion_num_res_blocks", 3),
                "modality_dropout": 0.0,
            },
        }
        architecture = ck.get("architecture", "")
        if architecture == "context_question_answer_frozen_encoder_fusion":
            self.model, specs, self.branch_ckpts = _build_context_question_answer_frozen_fusion(
                eval_config, embed_weights, device)
            self.dataset_factory = _context_dataset_from_encoder_ckpt
            self.wrapper_cls = ContextQuestionAnswerFrozenFusionDataset
        else:
            self.model, specs, self.branch_ckpts = _build_question_answer_frozen_fusion(
                eval_config, embed_weights, device)
            self.dataset_factory = _question_dataset_from_encoder_ckpt
            self.wrapper_cls = QuestionAnswerFrozenFusionDataset
        missing, unexpected = self.model.load_state_dict(ck["state_dict"], strict=False)
        bad = [k for k in missing if not k.startswith("encoders.")]
        if bad or unexpected:
            raise RuntimeError(f"{mode} fusion head mismatch: missing={bad} unexpected={list(unexpected)}")
        self.model.eval()
        self.branch_names = [s["name"] for s in specs]
        log(f"[{mode}] fusion probe loaded: {checkpoint_path} branches={self.branch_names} "
            f"target={self.binary_target_name}")


class ProbeScorer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        gpu_id = config.get("gpu_id")
        if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config["model_name"]
        feature_cfg = config.get("feature", {})
        torch_dtype = _dtype_from_string(feature_cfg.get("torch_dtype", "bfloat16"))
        # eager mandatory: training features (attention maps) were extracted with eager
        attn_impl = feature_cfg.get("attn_implementation", "eager")
        self.hidden_layer = int(feature_cfg.get("hidden_layer", 17))
        # attention-map layer subset; must match training features (gemma: full-attn
        # layers only). None = all layers (qwen behaviour).
        al = feature_cfg.get("attention_layers")
        self.attention_layers = [int(x) for x in al] if al else None

        log(f"Loading feature-extraction LLM: {self.model_name} (dtype={torch_dtype}, attn={attn_impl})")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForImageTextToText.from_pretrained(
            self.model_name, torch_dtype=torch_dtype, attn_implementation=attn_impl,
            device_map=feature_cfg.get("device_map", "cuda"),
        ).eval()
        embed_weights = self.llm.get_input_embeddings().weight.detach().float().cpu()

        self.stacks: dict[str, _FusionStack] = {}
        for mode, key in (("context", "context_fusion_checkpoint"),
                          ("question", "question_fusion_checkpoint")):
            path = config.get(key)
            if path:
                self.stacks[mode] = _FusionStack(mode, path, self.model_name, embed_weights, self.device)
        if not self.stacks:
            raise ValueError("configure context_fusion_checkpoint and/or question_fusion_checkpoint")
        self.tmp_root = Path(config.get("tmp_dir") or (Path(tempfile.gettempdir()) / "ragprobe_scorer"))
        self.tmp_root.mkdir(parents=True, exist_ok=True)

        # dual-branch (q-only + with-context) late fusion, used by the two-phase flow.
        self.dual_head: _DualHead | None = None
        dual_ckpt = config.get("dual_fusion_checkpoint")
        if dual_ckpt:
            if "question" not in self.stacks or "context" not in self.stacks:
                raise ValueError("dual_fusion_checkpoint needs BOTH question and context stacks")
            self.dual_head = _DualHead(dual_ckpt, self.device)
            log(f"[dual] q+ctx fusion head loaded: {dual_ckpt} "
                f"ctx_branches={self.dual_head.ctx_branches} qonly_branches={self.dual_head.qonly_branches}")
        cc = config.get("qonly_cache", {})
        self.qcache = _QOnlyCache(max_entries=int(cc.get("max_entries", 20000)),
                                  ttl_s=float(cc.get("ttl_s", 1800.0)))
        # Serializes every model-touching section. Mandatory: /score_question and
        # /score_context are sync endpoints run in FastAPI's threadpool (and the
        # micro-batcher worker is yet another thread), while feature extraction
        # registers TEMPORARY forward hooks on the SHARED LLM
        # (_run_with_residual_hooks) — concurrent requests would fire each other's
        # hooks and silently cross-contaminate residual features. The GPU is a
        # serial resource anyway, so the lock costs queueing, not throughput.
        self._infer_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def score(self, item: dict[str, Any]) -> dict[str, Any]:
        return self.score_many([item])[0]

    def score_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._score_one_safe(it) for it in items]

    def _score_one_safe(self, item: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._score_one(item)
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            return {"example_id": item.get("example_id"), "status": "failed", "error": str(exc)}

    def _resolve_mode(self, item: dict[str, Any]) -> str:
        mode = item.get("mode")
        if mode not in ("context", "question"):
            mode = "context" if (item.get("contexts") or item.get("context")) else "question"
        if mode not in self.stacks:
            raise ValueError(f"mode '{mode}' has no fusion checkpoint configured")
        return mode

    def _score_one(self, item: dict[str, Any], _return_vecs: bool = False):
        started = time.monotonic()
        mode = self._resolve_mode(item)
        stack = self.stacks[mode]
        params = MODE_DEFAULTS[mode]
        question = str(item.get("question") or "").strip()
        if not question:
            raise ValueError("'question' is required")
        contexts = _normalize_contexts(item.get("contexts", item.get("context")))
        ex_id = str(item.get("example_id") or uuid.uuid4().hex[:12])
        record = {"example": {"id": ex_id, "question": question, "contexts": contexts}}

        with self._infer_lock, tempfile.TemporaryDirectory(dir=self.tmp_root) as td:
            tdir = Path(td)
            row = _extract_one_unified(
                self.llm, self.processor, record, self.model_name,
                tensor_path=tdir / "t.pt", token_ids_path=tdir / "k.pt", sidecar_path=tdir / "s.pt",
                target_type=mode, max_len=params["max_len"],
                answer_tokens=int(item.get("max_answer_tokens", params["answer_tokens"])),
                prob_top_k=params["prob_top_k"],
                answer_override=item.get("answer"),
                hidden_layer=self.hidden_layer, add_resid_lens=True,
                attention_layers=self.attention_layers,
            )
            fx = row.get("feature_extraction", {})
            if fx.get("status") != "success":
                failed = {"example_id": item.get("example_id"), "mode": mode, "status": "failed",
                          "error": f"feature_extraction:{fx.get('reason') or fx.get('status')}"}
                return (failed, None) if _return_vecs else failed
            answer_text = fx["rag"]["spans"].get("answer_text", "")

            meta = [row]
            branch_ds = {
                name: stack.dataset_factory(meta, self.tokenizer, stack.branch_ckpts[name],
                                            stack.positive_transition, stack.label_mode)
                for name in stack.branch_names
            }
            loader = DataLoader(stack.wrapper_cls(branch_ds), batch_size=1, shuffle=False, num_workers=0)
            branch_batches, _targets, _idx = next(iter(loader))
            branch_batches = _move_branch_batches(branch_batches, self.device)
            with torch.no_grad():
                # per-branch pooled vecs (needed for dual q+ctx fusion); kept on CPU
                branch_vecs = {n: stack.model.encoders[n].encode(*branch_batches[n]).reshape(-1).cpu()
                               for n in stack.branch_names} if _return_vecs else None
                out = stack.model(branch_batches, return_multitask=True)
            p = float(torch.sigmoid(out["binary_logits"]).reshape(-1)[0])
            branch_probs = torch.sigmoid(out["modality_logits"]).reshape(-1).tolist()

        result = {
            "example_id": item.get("example_id"),
            "mode": mode,
            "answer": answer_text,
            "answer_source": "provided" if (item.get("answer") or "").strip() else "generated",
            "p_correct": p,
            "binary_target": stack.binary_target_name,   # what p_correct estimates
            "branch_probs": {n: float(v) for n, v in zip(stack.branch_names, branch_probs)},
            "probe_checkpoint": stack.checkpoint_path,
            "timing_s": round(time.monotonic() - started, 3),
            "status": "success",
        }
        return (result, branch_vecs) if _return_vecs else result

    # ------------------------------------------------------------------ #
    # two-phase RAG flow: q-only first (cached, returns uuid) -> with-ctx (reuses it)
    # ------------------------------------------------------------------ #
    def score_question(self, item: dict[str, Any]) -> dict[str, Any]:
        """Phase 1. Score the question closed-book, cache its q-only branch vecs under a
        fresh uuid, and return that uuid for the subsequent with-context call(s)."""
        it = {**item, "mode": "question"}
        try:
            result, vecs = self._score_one(it, _return_vecs=True)
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            return {"example_id": item.get("example_id"), "status": "failed", "error": str(exc)}
        if result.get("status") == "success" and vecs is not None and self.dual_head is not None:
            key = self.qcache.put({"vecs": vecs, "question": it.get("question"),
                                   "p_correct": result["p_correct"]})
            result["qonly_uuid"] = key
            result["qonly_cache"] = self.qcache.stats()
        elif self.dual_head is None:
            result["qonly_uuid"] = None
            result["note"] = "no dual_fusion_checkpoint configured; nothing cached"
        return result

    def score_context(self, item: dict[str, Any]) -> dict[str, Any]:
        """Phase 2. Score question+context. If `qonly_uuid` resolves to a cached q-only
        entry AND a dual head is loaded, fuse (q-only ‖ with-context) into `p_correct`;
        else fall back to the context-only probe. Safe to call many times per uuid."""
        it = {**item, "mode": "context"}
        key = item.get("qonly_uuid")
        try:
            result, ctx_vecs = self._score_one(it, _return_vecs=True)
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            return {"example_id": item.get("example_id"), "status": "failed", "error": str(exc)}
        if result.get("status") != "success":
            return result
        result["p_correct_context_only"] = result["p_correct"]
        if self.dual_head is None:
            result["fusion"] = "context_only (no dual head configured)"
            return result
        cached = self.qcache.get(key) if key else None
        if cached is None:
            result["fusion"] = "context_only" + (" (uuid miss/expired)" if key else " (no qonly_uuid)")
            result["qonly_uuid"] = key
            return result
        p_fused = self.dual_head.fuse(ctx_vecs, cached["vecs"])
        result["p_correct"] = p_fused                 # headline = dual-branch fused score
        result["p_correct_fused"] = p_fused
        result["p_correct_qonly"] = cached.get("p_correct")
        result["fusion"] = "dual_qctx"
        result["qonly_uuid"] = key
        result["probe_checkpoint"] = f"{result['probe_checkpoint']} + {self.dual_head.ckpt_path}"
        return result
