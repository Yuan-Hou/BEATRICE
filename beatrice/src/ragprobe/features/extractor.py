from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from ragprobe.prompts.qa import LEGACY_MUSIQUE_FEATURE_PROMPT, QUESTION_ONLY_FEATURE_PROMPT
from ragprobe.utils.io import atomic_write_jsonl, read_jsonl
from ragprobe.utils.scope import filter_records_by_scope, load_scope_ids, scope_path_from_config
from ragprobe.utils.logging import log


EXTRACTOR_VERSION = "unified_answer_span_v2"
SCALAR_FEATURES = ("p", "entropy", "surprisal", "top1_prob", "margin")


def _dtype_from_string(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _get_example(record: dict) -> dict:
    if "example" in record:
        return record["example"]

    raw_contexts = record.get("contexts", record.get("context", []))
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

    raw_answer = record.get("answers", record.get("answer", []))
    answers = raw_answer if isinstance(raw_answer, list) else ([] if raw_answer is None else [str(raw_answer)])
    return {
        "id": str(record.get("id", record.get("question", ""))),
        "dataset": record.get("dataset", "legacy"),
        "split": record.get("split", record.get("set", "unknown")),
        "question": record["question"],
        "contexts": contexts,
        "answers": answers,
        "answerable": record.get("answerable", True),
        "metadata": record.get("metadata", {}),
    }


def _get_example_id(record: dict) -> str:
    example = _get_example(record)
    return str(example.get("id") or example.get("question") or "")


def _format_context_legacy(record: dict) -> str:
    if "example" not in record and "context" in record:
        raw = record.get("context") or []
        if isinstance(raw, str):
            return raw
        return "\n".join("- " + str(x) for x in raw)
    example = _get_example(record)
    rows = []
    for ctx in example.get("contexts") or []:
        if isinstance(ctx, str):
            text = ctx
        else:
            text = ctx.get("text") or ctx.get("paragraph_text") or ""
        rows.append("- " + str(text))
    return "\n".join(rows)


def _get_tensor_path(record: dict) -> Optional[str]:
    return record.get("tensor_path") or record.get("feature_extraction", {}).get("rag", {}).get("tensor_path")


def _is_reusable_record(record: dict, retry_failed: bool = False) -> bool:
    fe = record.get("feature_extraction", {})
    status = fe.get("status")
    rag = fe.get("rag", {})
    tensor_path = _get_tensor_path(record)
    if status == "success":
        return rag.get("extractor_version") == EXTRACTOR_VERSION and bool(tensor_path) and Path(tensor_path).exists()
    if status == "failed":
        return not retry_failed and fe.get("extractor_version") == EXTRACTOR_VERSION
    return False


def _find_token_span(tokenizer, input_ids, target_tokens, min_start: int = 0):
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(target_tokens, "tolist"):
        target_tokens = target_tokens.tolist()
    target_tokens = [int(x) for x in target_tokens if int(x) >= 0]
    if not target_tokens:
        return 0, -1
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids)
    target_tokens_list = tokenizer.convert_ids_to_tokens(target_tokens)
    for i in range(max(0, int(min_start)), len(tokens_list) - len(target_tokens_list) + 1):
        if tokens_list[i:i + len(target_tokens_list)] == target_tokens_list:
            return i, i + len(target_tokens_list)
    return 0, -1


def _find_text_span(tokenizer, input_ids, text: str, min_start: int = 0):
    text = (text or "").strip()
    if not text:
        return 0, -1
    variants = [
        (text, 0),
        (" " + text, 0),
        ("\n" + text, 0),
        ("Question: " + text, len(tokenizer("Question:", add_special_tokens=False)["input_ids"])),
    ]
    seen = set()
    for variant, trim_prefix_tokens in variants:
        if variant in seen:
            continue
        seen.add(variant)
        target = tokenizer(variant, add_special_tokens=False)["input_ids"]
        start, end = _find_token_span(tokenizer, input_ids, target, min_start=min_start)
        if start != 0 and end > start:
            return start + trim_prefix_tokens, end
    return 0, -1


def _build_failed_record(record: dict, reason: str) -> dict:
    out = dict(record)
    out["tensor_path"] = None
    out["feature_extraction"] = {
        "status": "failed",
        "mode": EXTRACTOR_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "reason": reason,
    }
    return out


def _add_transition_if_needed(out: dict) -> dict:
    if "derived" not in out and ("is_correct_without_context" in out or "is_correct_with_context" in out):
        base_corr = bool(out.get("is_correct_without_context", False))
        rag_corr = bool(out.get("is_correct_with_context", False))
        transition = ("Y" if base_corr else "N") + ("Y" if rag_corr else "N")
        out["derived"] = {
            "transition": transition,
            "is_improved": transition == "NY",
            "is_regressed": transition == "YN",
            "is_always_correct": transition == "YY",
            "is_always_incorrect": transition == "NN",
        }
    return out


def _build_success_record(
    record: dict,
    tensor_path: Path,
    token_ids_path: Path,
    sidecar_path: Path,
    model_name: str,
    target_type: str,
    feature_names: list[str],
    spans: dict,
    max_len: int,
) -> dict:
    out = dict(record)
    out["tensor_path"] = str(tensor_path)
    out["feature_extraction"] = {
        "status": "success",
        "model_name": model_name,
        "mode": EXTRACTOR_VERSION,
        "rag": {
            "tensor_path": str(tensor_path),
            "token_ids_path": str(token_ids_path),
            "sidecar_path": str(sidecar_path),
            "target_type": target_type,
            "extractor_version": EXTRACTOR_VERSION,
            "tensor_schema": "packed_token_map",
            "token_axis": "context_question_answer" if target_type != "question" else "question_answer",
            "stored_len": int(max_len),
            "num_feature_channels": len(feature_names),
            "feature_names": feature_names,
            "spans": spans,
            "sidecar_schema": {
                "topk": "top-k next-token distribution for each packed token",
                "attention_maps": "layer/head maps for answer->question, answer->context, question->context",
                "last_hidden_states": "fp16 final-layer token features split by context/question/answer",
            },
        },
    }
    return _add_transition_if_needed(out)


def _is_fatal_cuda_error(exc: Exception) -> bool:
    message = repr(exc).lower()
    return any(x in message for x in ["cuda error", "acceleratorerror", "device-side assert", "cublas", "cudnn"])


GEMMA_TURN_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def _apply_chat_template(processor, prompt: str, device):
    tokenizer = processor.tokenizer
    if getattr(tokenizer, "chat_template", None) is None:
        # gemma-4-E4B ships without a chat template; use the same turn format as
        # label construction (openai_client PROMPT_STYLES["gemma_turns"]).
        # The turn delimiters are PLAIN TEXT on this tokenizer, so content and
        # suffix are tokenized separately: otherwise the content's last token
        # merges with '<' of <end_of_turn> (e.g. ')<') and teacher-forced answer
        # spans become unfindable. tokenizer() adds BOS on the first chunk only.
        pre = tokenizer("<start_of_turn>user\n" + prompt, return_tensors="pt")
        suf = tokenizer("<end_of_turn>\n<start_of_turn>model\n",
                        add_special_tokens=False, return_tensors="pt")
        input_ids = torch.cat([pre["input_ids"], suf["input_ids"]], dim=1)
        return {"input_ids": input_ids.to(device),
                "attention_mask": torch.ones_like(input_ids).to(device)}
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device)


TURN_STOP_STRINGS = ("<end_of_turn>", "<start_of_turn>")


def _clean_generated_answer(tokenizer, answer_ids):
    """Cut a generated continuation at eos and at turn-delimiter STRINGS.

    gemma-4-E4B's tokenizer has no <end_of_turn> special token — the model emits
    the delimiter as ordinary text (7 tokens), so id-level eos cuts miss it and
    the tail contains a fabricated next turn. Cut at the decoded string level
    and re-encode. Returns (clean_ids, clean_text)."""
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        eos_positions = (answer_ids == eos_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            answer_ids = answer_ids[:int(eos_positions[0].item())]
    text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    cut = min([text.find(s) for s in TURN_STOP_STRINGS if s in text], default=-1)
    if cut >= 0:
        text = text[:cut]
        answer_ids = torch.tensor(
            tokenizer(text.strip(), add_special_tokens=False)["input_ids"], dtype=torch.long)
    return answer_ids, text.strip()


def _generate_answer_ids(model, processor, prompt: str, max_new_tokens: int):
    inputs = _apply_chat_template(processor, prompt, model.device)
    input_len = int(inputs["input_ids"].shape[1])
    stop_kwargs = {}
    if getattr(processor.tokenizer, "chat_template", None) is None:
        # turn delimiters are plain text on this tokenizer; stop generation on them
        stop_kwargs = {"stop_strings": list(TURN_STOP_STRINGS), "tokenizer": processor.tokenizer}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            **stop_kwargs,
        )
    answer_ids = generated[0, input_len:input_len + int(max_new_tokens)].detach()
    answer_ids, _ = _clean_generated_answer(processor.tokenizer, answer_ids)
    return answer_ids


def _decoder_layers(model):
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ]
    for obj in candidates:
        layers = getattr(obj, "layers", None) if obj is not None else None
        if layers is not None:
            return list(layers)
    for name, module in model.named_modules():
        if name.endswith("layers") and hasattr(module, "__iter__"):
            try:
                layers = list(module)
            except TypeError:
                continue
            if layers:
                return layers
    return []


def _as_tensor_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _is_postnorm_layer(layer) -> bool:
    """gemma2/3/4-style block: contributions are re-normalized before the residual
    add (h += post_*_layernorm(module(...))), so the true additive contribution is
    the POST-norm output, not the raw module output. llama/qwen lack
    post_feedforward_layernorm and keep the legacy raw-output hooks."""
    return getattr(layer, "post_feedforward_layernorm", None) is not None


def _run_with_residual_hooks(model, processor, prompt: str):
    residuals: dict[str, torch.Tensor] = {}
    handles = []
    for layer_idx, layer in enumerate(_decoder_layers(model)):
        if _is_postnorm_layer(layer):
            attn = layer.post_attention_layernorm
            mlp = layer.post_feedforward_layernorm
        else:
            attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
            mlp = getattr(layer, "mlp", None)
        if attn is not None:
            def make_attn_hook(i):
                def hook(_module, _inputs, output):
                    value = _as_tensor_output(output)
                    if value is not None:
                        residuals[f"resid_attn_l{i:02d}"] = value.detach().float().cpu()
                return hook
            handles.append(attn.register_forward_hook(make_attn_hook(layer_idx)))
        if mlp is not None:
            def make_mlp_hook(i):
                def hook(_module, _inputs, output):
                    residuals[f"resid_mlp_l{i:02d}"] = _as_tensor_output(output).detach().float().cpu()
                return hook
            handles.append(mlp.register_forward_hook(make_mlp_hook(layer_idx)))
    try:
        inputs = _apply_chat_template(processor, prompt, model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    return inputs, outputs, residuals


def _token_distribution(outputs, token_ids: torch.Tensor, positions: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
    positions = positions.to(token_ids.device).long()
    valid = positions > 0
    safe_positions = positions.clamp_min(1)
    logits = outputs.logits[0, safe_positions - 1].detach().float()
    probs = torch.softmax(logits, dim=-1)
    actual = token_ids[safe_positions].unsqueeze(1)
    actual_p = probs.gather(1, actual).squeeze(1).clamp_min(1e-10)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    top_vals, top_ids = torch.topk(probs, k=min(int(top_k), probs.shape[-1]), dim=-1)
    top1 = top_vals[:, 0]
    margin = top_vals[:, 0] - top_vals[:, 1] if top_vals.shape[1] > 1 else top_vals[:, 0]
    for tensor in [actual_p, entropy, top1, margin]:
        tensor[~valid] = 0.0
    surprisal = -torch.log(actual_p.clamp_min(1e-10))
    surprisal[~valid] = 0.0
    return {
        "p": actual_p.cpu(),
        "entropy": entropy.cpu(),
        "surprisal": surprisal.cpu(),
        "top1_prob": top1.cpu(),
        "margin": margin.cpu(),
        "topk_token_ids": top_ids.cpu().to(torch.long),
        "topk_probs": top_vals.cpu(),
    }


def _span_mask(max_len: int, start: int, end: int) -> torch.Tensor:
    out = torch.zeros(max_len, dtype=torch.float32)
    if start >= 0 and end > start:
        out[start:min(end, max_len)] = 1.0
    return out


def _slice_positions(start: int, length: int) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(int(start), int(start + length), dtype=torch.long)


def _pack_spans(max_len: int, spans: dict[str, tuple[int, int]], target_type: str):
    order = ["question", "answer"] if target_type == "question" else ["context", "question", "answer"]
    packed_positions = []
    packed = {}
    cursor = 0
    for name in order:
        start, end = spans.get(name, (0, -1))
        length = max(0, int(end) - int(start))
        if name == "answer":
            length = min(length, max(0, max_len - cursor))
        elif name == "question":
            answer_len = max(0, spans.get("answer", (0, -1))[1] - spans.get("answer", (0, -1))[0])
            length = min(length, max(0, max_len - cursor - min(answer_len, 32)))
        else:
            q_len = max(0, spans.get("question", (0, -1))[1] - spans.get("question", (0, -1))[0])
            a_len = min(max(0, spans.get("answer", (0, -1))[1] - spans.get("answer", (0, -1))[0]), 32)
            length = min(length, max(0, max_len - cursor - q_len - a_len))
        packed[name] = {"start": cursor, "end": cursor + length, "len": length, "source_start": int(start), "source_end": int(start + length)}
        packed_positions.append(_slice_positions(start, length))
        cursor += length
    positions = torch.cat(packed_positions, dim=0) if packed_positions else torch.empty(0, dtype=torch.long)
    if positions.numel() < max_len:
        positions = F.pad(positions, (0, max_len - positions.numel()), value=0)
    return packed, positions[:max_len]



def _last_hidden_state_segments(outputs, packed: dict[str, dict], target_type: str, layer: int | None = None) -> dict[str, torch.Tensor]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        names = ["question", "answer"] if target_type == "question" else ["context", "question", "answer"]
        return {name: torch.empty(0, 0, dtype=torch.float16) for name in names}
    last_hidden = hidden_states[-1 if layer is None else layer][0].detach().cpu().to(torch.float16)
    out = {}
    for name in (["question", "answer"] if target_type == "question" else ["context", "question", "answer"]):
        span = packed.get(name, {}) or {}
        start = max(0, int(span.get("source_start", 0) or 0))
        length = max(0, int(span.get("len", 0) or 0))
        end = max(start, min(start + length, int(last_hidden.shape[0])))
        out[name] = last_hidden[start:end].contiguous()
    return out

def _attention_map(outputs, query_positions: torch.Tensor, key_positions: torch.Tensor,
                   attention_layers: list[int] | None = None) -> torch.Tensor:
    attentions = list(outputs.attentions or [])
    if attention_layers is not None:
        # models with interleaved sliding/full attention (gemma4) return maps for
        # every layer; keep only the requested (full-attention) layers — mirrors
        # Qwen3.5 where output_attentions natively yields just the 8 full layers.
        attentions = [attentions[i] for i in attention_layers if 0 <= i < len(attentions)]
    if query_positions.numel() == 0 or key_positions.numel() == 0:
        layers = len(attentions)
        heads = int(attentions[0].shape[1]) if attentions else 0
        return torch.empty(layers, heads, int(query_positions.numel()), int(key_positions.numel()), dtype=torch.float16)
    maps = []
    q = query_positions.to(torch.long)
    k = key_positions.to(torch.long)
    for attn in attentions:
        # [batch, heads, query, key]
        maps.append(attn[0].detach().float().cpu()[:, q][:, :, k])
    return torch.stack(maps, dim=0).half()


def _residual_norm_channels(residuals: dict[str, torch.Tensor], packed_positions: torch.Tensor, max_len: int):
    names = []
    channels = []
    source_positions = packed_positions.long()
    valid = source_positions > 0
    for name in sorted(residuals):
        value = residuals[name][0]
        safe_pos = source_positions.clamp_min(0).clamp_max(value.shape[0] - 1)
        norms = torch.linalg.vector_norm(value[safe_pos], dim=-1).float()
        norms[~valid] = 0.0
        names.append(name + "_norm")
        channels.append(norms[:max_len].cpu())
    return names, channels


def _build_prompt(question: str, context: str | None) -> tuple[str, str]:
    if context is None:
        return "question", QUESTION_ONLY_FEATURE_PROMPT.substitute(question=question)
    return "context", LEGACY_MUSIQUE_FEATURE_PROMPT.substitute(context=context, question=question)


def _extract_one_unified(
    model,
    processor,
    record: dict,
    model_name: str,
    tensor_path: Path,
    token_ids_path: Path,
    sidecar_path: Path,
    target_type: str,
    max_len: int,
    answer_tokens: int,
    prob_top_k: int,
    answer_override: str | None = None,
    hidden_layer: int | None = None,
    add_resid_lens: bool = False,
    attention_layers: list[int] | None = None,
):
    """One-record feature extraction.

    Online-serving extensions (all default-off; offline pipelines unchanged):
      answer_override — score a caller-provided answer instead of generating one;
      hidden_layer    — store this layer's hidden segments in the sidecar instead of the
                        last layer (training uses L17, added offline by
                        scripts/extract_withctx_L17rd.py — this computes it inline);
      add_resid_lens  — append the 64 `resid_*_lens` logit-lens direction channels
                        (same math as scripts/extract_withctx_L17rd.py) to the tensor.
    """
    example = _get_example(record)
    question = str(example.get("question") or "").strip()
    context = None if target_type == "question" else _format_context_legacy(record).strip()
    if not question:
        return _build_failed_record(record, "empty_question")
    if target_type != "question" and not context:
        return _build_failed_record(record, "empty_context")

    _, base_prompt = _build_prompt(question, context)
    if answer_override is not None and str(answer_override).strip():
        answer_text = str(answer_override).strip()
        answer_ids = torch.tensor(
            processor.tokenizer(answer_text, add_special_tokens=False)["input_ids"],
            dtype=torch.long,
        )
    else:
        answer_ids = _generate_answer_ids(model, processor, base_prompt, answer_tokens)
        answer_text = processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    if not answer_text:
        return _build_failed_record(record, "empty_generated_answer")

    combined_prompt = base_prompt + " " + answer_text
    inputs, outputs, residuals = _run_with_residual_hooks(model, processor, combined_prompt)
    ids = inputs["input_ids"][0]

    spans: dict[str, tuple[int, int]] = {}
    if context is not None:
        spans["context"] = _find_text_span(processor.tokenizer, ids, context)
        if spans["context"][0] == 0 or spans["context"][1] <= spans["context"][0]:
            return _build_failed_record(record, "context_not_found")
    q_start, q_end = _find_text_span(processor.tokenizer, ids, question)
    if q_start == 0 or q_end <= q_start:
        return _build_failed_record(record, "question_not_found")
    spans["question"] = (q_start, q_end)

    a_start, a_end = _find_text_span(processor.tokenizer, ids, answer_text, min_start=q_end)
    if a_start == 0 or a_end <= a_start:
        a_start, a_end = _find_token_span(processor.tokenizer, ids, answer_ids, min_start=q_end)
    if a_start == 0 or a_end <= a_start:
        return _build_failed_record(record, "answer_not_found")
    spans["answer"] = (a_start, min(a_end, a_start + int(answer_tokens)))

    packed, packed_positions = _pack_spans(max_len, spans, target_type)
    if packed_positions.numel() == 0 or int((packed_positions > 0).sum().item()) == 0:
        return _build_failed_record(record, "empty_packed_positions")

    token_stats = _token_distribution(outputs, ids, packed_positions.to(ids.device), top_k=prob_top_k)
    feature_names: list[str] = []
    channels: list[torch.Tensor] = []
    for name in SCALAR_FEATURES:
        feature_names.append(name)
        channels.append(token_stats[name].float()[:max_len])

    residual_names, residual_channels = _residual_norm_channels(residuals, packed_positions, max_len)
    feature_names.extend(residual_names)
    channels.extend(residual_channels)

    for name in (["context", "question", "answer"] if target_type != "question" else ["question", "answer"]):
        span = packed.get(name, {"start": 0, "end": 0})
        feature_names.append(f"mask_{name}_tokens")
        channels.append(_span_mask(max_len, int(span["start"]), int(span["end"])))
    feature_names.append("position_frac")
    channels.append(torch.arange(max_len, dtype=torch.float32) / max(float(max_len - 1), 1.0))

    if add_resid_lens:
        # logit-lens push channels, identical math to scripts/extract_withctx_L17rd.py:
        # per layer, <W_U[next], N(h+contribution)> - <W_U[next], N(h)> gathered at the
        # packed source positions. Appended AFTER the base channels (same order as the
        # offline *_L17_rd tensors).
        lm = None
        for cand in (getattr(getattr(model, "model", None), "language_model", None),
                     getattr(model, "language_model", None),
                     getattr(model, "model", None)):
            if cand is not None and hasattr(cand, "norm") and hasattr(cand, "layers"):
                lm = cand
                break
        if lm is None:
            raise RuntimeError("cannot locate language model (norm/layers) for resid lens")
        norm_mod = lm.norm
        W_U = model.lm_head.weight
        hs = outputs.hidden_states
        seq = int(ids.shape[0])
        nxt = torch.empty(seq, dtype=torch.long, device=ids.device)
        nxt[:-1] = ids[1:]
        nxt[-1] = ids[-1]
        U = W_U[nxt].float()
        has_next = torch.ones(seq, device=ids.device)
        has_next[-1] = 0.0

        def _lens(vec):
            return (norm_mod(vec).float() * U).sum(-1)

        postnorm = _is_postnorm_layer(lm.layers[0])
        lens_names, lens_channels = [], []
        for l in range(len(lm.layers)):
            a = residuals.get(f"resid_attn_l{l:02d}")
            m = residuals.get(f"resid_mlp_l{l:02d}")
            hsl, hsl1 = hs[l][0], hs[l + 1][0]
            if a is not None:
                a = a[0].to(hsl.device)
                lens_names.append(f"resid_attn_l{l:02d}_lens")
                lens_channels.append((_lens(hsl + a) - _lens(hsl)) * has_next)
            if m is not None:
                m = m[0].to(hsl1.device)
                lens_names.append(f"resid_mlp_l{l:02d}_lens")
                if postnorm:
                    # gemma: h_{l+1} = scalar*(h_l + An + Mn + PLE); hsl1 - Mn is not a
                    # real state, but h_l + An (+ Mn) are the exact intermediate streams
                    base = hsl + a if a is not None else hsl
                    lens_channels.append((_lens(base + m) - _lens(base)) * has_next)
                else:
                    lens_channels.append((_lens(hsl1) - _lens(hsl1 - m)) * has_next)
        valid = (packed_positions > 0).float().to(lens_channels[0].device)
        spos = packed_positions.long().clamp(0, seq - 1).to(lens_channels[0].device)
        for name, ch in zip(lens_names, lens_channels):
            feature_names.append(name)
            channels.append((ch[spos] * valid).float().cpu())

    features = torch.stack([ch.float().cpu() for ch in channels], dim=0)
    torch.save(features.half().cpu(), tensor_path)

    safe_positions = packed_positions.long().clamp_min(0).clamp_max(ids.shape[0] - 1).to(ids.device)
    packed_token_ids = ids[safe_positions].detach().cpu().to(torch.long)
    packed_token_ids[packed_positions <= 0] = processor.tokenizer.pad_token_id or 0
    torch.save(packed_token_ids, token_ids_path)

    q_pos = _slice_positions(packed["question"]["source_start"], packed["question"]["len"])
    a_pos = _slice_positions(packed["answer"]["source_start"], packed["answer"]["len"])
    c_pos = _slice_positions(packed.get("context", {}).get("source_start", 0), packed.get("context", {}).get("len", 0))
    sidecar = {
        "schema_version": EXTRACTOR_VERSION,
        "target_type": target_type,
        "answer_text": answer_text,
        "packed_token_ids": packed_token_ids,
        "source_positions": packed_positions.cpu().to(torch.long),
        "topk_token_ids": token_stats["topk_token_ids"].to(torch.long),
        "topk_probs": token_stats["topk_probs"].half(),
        "attention_maps": {
            "answer_to_question": _attention_map(outputs, a_pos, q_pos, attention_layers),
            "answer_to_context": _attention_map(outputs, a_pos, c_pos, attention_layers),
            "question_to_context": _attention_map(outputs, q_pos, c_pos, attention_layers),
        },
        "last_hidden_states": _last_hidden_state_segments(outputs, packed, target_type, layer=hidden_layer),
    }
    if hidden_layer is not None:
        sidecar["hidden_layer"] = int(hidden_layer)
    torch.save(sidecar, sidecar_path)

    meta_spans = {
        "packed": packed,
        "source_prompt": {
            name: {"start": int(start), "end": int(end), "len": int(max(0, end - start))}
            for name, (start, end) in spans.items()
        },
        "seq_len": int(ids.shape[0]),
        "answer_text": answer_text,
        "prob_top_k": int(prob_top_k),
    }
    return _build_success_record(record, tensor_path, token_ids_path, sidecar_path, model_name, target_type, feature_names, meta_spans, max_len)


def _normalize_target_type(value: str) -> str:
    value = str(value or "context").lower()
    if value in {"question", "question_only", "qa"}:
        return "question"
    if value in {"context", "with_context", "context_question_answer", "cqa", "contrastive_token_map"}:
        return "context"
    raise ValueError(f"Unsupported feature target_type: {value}")


def run_feature_extraction(config: dict):
    # pure inference; without this the add_resid_lens channels (built from W_U and
    # norm, both model params) carry autograd graphs into the saved tensors
    # (mirrors scripts/extract_withctx_L17rd.py's global switch)
    torch.set_grad_enabled(False)
    gpu_id = config.get("gpu_id")
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model_name = config["model_name"]
    input_path = config["input_result_path"]
    output_meta_path = config["output"]["meta_path"]
    tensor_dir = Path(config["output"]["tensor_dir"])
    token_dir = Path(config["output"].get("token_ids_dir", str(tensor_dir) + "_token_ids"))
    sidecar_dir = Path(config["output"].get("sidecar_dir", str(tensor_dir) + "_sidecars"))
    tensor_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    feature_cfg = config.get("feature", {})
    target_type = _normalize_target_type(feature_cfg.get("target_type", "context"))
    max_len = int(feature_cfg.get("max_context_len", config.get("max_context_len", 1024)))
    answer_tokens = int(feature_cfg.get("answer_tokens", 32))
    prob_top_k = int(feature_cfg.get("prob_top_k", 20))
    torch_dtype = _dtype_from_string(feature_cfg.get("torch_dtype", "bfloat16"))
    attn_impl = feature_cfg.get("attn_implementation", "eager")
    hidden_layer = feature_cfg.get("hidden_layer")
    hidden_layer = int(hidden_layer) if hidden_layer is not None else None
    add_resid_lens = bool(feature_cfg.get("add_resid_lens", False))
    attention_layers = feature_cfg.get("attention_layers")
    attention_layers = [int(x) for x in attention_layers] if attention_layers else None
    # answer_source:
    #   "generate" (default, paper protocol) — training-format greedy regeneration.
    #   "record" — teacher-force the stored label-time answer for every row.
    #   "generate_or_record" — generate; if the greedy answer comes out empty
    #     (gemma-4-E4B emits an immediate <end_of_turn> on some rows), fall back to
    #     the stored answer instead of dropping the row. Keeps eval-set coverage
    #     complete while leaving the protocol untouched for non-empty rows; the
    #     fallback mirrors the deployed scorer service (answer_override semantics).
    #   "file" — teacher-force answers from feature.answer_file, a jsonl of
    #     {id, answers: {with_context, without_context}} rows (e.g. the vLLM
    #     pre-generated training-format answers, scripts/gemma/gen_trainfmt_answers_vllm.py).
    # Provenance is recorded per row as feature_extraction.rag.answer_source_used.
    answer_source = str(feature_cfg.get("answer_source", "generate"))
    if answer_source not in ("generate", "record", "generate_or_record", "file"):
        raise ValueError(f"Unsupported feature.answer_source: {answer_source}")
    answer_run_key = "with_context" if target_type == "context" else "without_context"
    answer_file_map = None
    if answer_source == "file":
        answer_file = feature_cfg.get("answer_file")
        if not answer_file:
            raise ValueError("feature.answer_source=file requires feature.answer_file")
        answer_file_map = {}
        with open(answer_file) as fh:
            for line in fh:
                row = json.loads(line)
                answer_file_map[str(row["id"])] = row.get("answers") or {}
        log(f"Loaded {len(answer_file_map)} pre-generated answers from {answer_file}")

    max_examples = config.get("max_examples")
    resume = bool(config.get("resume", True))
    retry_failed = bool(config.get("retry_failed", False))
    save_every = int(config.get("save_every", config.get("save_every_batches", 100)))

    records = read_jsonl(input_path)
    scope_path = scope_path_from_config(config, "feature")
    if scope_path:
        before = len(records)
        records = filter_records_by_scope(records, load_scope_ids(scope_path))
        log(f"Applied fixed sample scope: {len(records)} / {before} records kept from {scope_path}")
    if max_examples:
        records = records[:int(max_examples)]

    existing_by_id = {}
    if resume and Path(output_meta_path).exists():
        existing_records = read_jsonl(output_meta_path)
        for old in existing_records:
            try:
                key = _get_example_id(old)
            except Exception:
                continue
            if _is_reusable_record(old, retry_failed=retry_failed):
                existing_by_id[key] = old
        log(f"Resume enabled. Reusable records: {len(existing_by_id)} / {len(existing_records)}")
    else:
        log("Resume disabled or no previous feature meta found.")

    pending_count = sum(1 for r in records if _get_example_id(r) not in existing_by_id)
    log(f"Input records: {len(records)}")
    log(f"Pending records: {pending_count}")
    if pending_count == 0:
        output_records = [existing_by_id[_get_example_id(r)] for r in records]
        atomic_write_jsonl(output_records, output_meta_path)
        log(f"Nothing to do. Saved normalized meta to: {output_meta_path}")
        return output_records

    log(
        "Loading Qwen model for unified feature extraction: "
        f"{model_name}; target_type={target_type}; max_len={max_len}; answer_tokens={answer_tokens}; prob_top_k={prob_top_k}"
    )
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
        ).to("cuda")
    except (ValueError, KeyError, OSError) as exc:
        log(f"AutoModelForImageTextToText load failed ({exc!r}); falling back to AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
        ).to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    output_records = []
    newly_extracted = 0
    failed = 0

    for idx, record in enumerate(tqdm(records, total=len(records), desc="Extracting unified token features")):
        key = _get_example_id(record)
        if key in existing_by_id:
            output_records.append(existing_by_id[key])
            continue

        safe_key = key.replace("/", "_").replace(" ", "_")[:160]
        tensor_path = tensor_dir / f"feature_{idx:08d}_{safe_key}.pt"
        token_ids_path = token_dir / f"tokens_{idx:08d}_{safe_key}.pt"
        sidecar_path = sidecar_dir / f"sidecar_{idx:08d}_{safe_key}.pt"

        def _record_answer() -> str | None:
            if answer_file_map is not None:
                text = str((answer_file_map.get(str(key)) or {}).get(answer_run_key) or "").strip() or None
            else:
                run = (record.get("runs") or {}).get(answer_run_key) or {}
                text = str(run.get("extracted_answer") or "").strip() or None
            if text is None:
                return None
            ov_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ov_ids) > answer_tokens:  # generated answers are capped here too
                text = processor.tokenizer.decode(
                    ov_ids[:answer_tokens], skip_special_tokens=True).strip()
            return text or None

        def _attempt(answer_override: str | None) -> dict:
            return _extract_one_unified(
                model=model,
                processor=processor,
                record=record,
                model_name=model_name,
                tensor_path=tensor_path,
                token_ids_path=token_ids_path,
                sidecar_path=sidecar_path,
                target_type=target_type,
                max_len=max_len,
                answer_tokens=answer_tokens,
                prob_top_k=prob_top_k,
                answer_override=answer_override,
                hidden_layer=hidden_layer,
                add_resid_lens=add_resid_lens,
                attention_layers=attention_layers,
            )

        try:
            source_used = None
            if answer_source in ("record", "file"):
                override = _record_answer()
                if override is None:
                    out = _build_failed_record(record, reason="missing_record_answer")
                else:
                    out = _attempt(override)
                    source_used = answer_source
            else:
                out = _attempt(None)
                source_used = "generate"
                if (answer_source == "generate_or_record"
                        and out.get("feature_extraction", {}).get("reason") == "empty_generated_answer"):
                    override = _record_answer()
                    if override is not None:
                        out = _attempt(override)
                        source_used = "record"
            if out.get("feature_extraction", {}).get("status") == "success":
                out["feature_extraction"].setdefault("rag", {})["answer_source_used"] = source_used
        except Exception as exc:
            log(f"Error extracting idx={idx}, id={key}: {exc}")
            if _is_fatal_cuda_error(exc):
                atomic_write_jsonl(output_records, output_meta_path)
                log("Fatal CUDA error detected. Saved completed records and stopping extraction.")
                raise
            out = _build_failed_record(record, reason=f"exception: {repr(exc)}")

        if out.get("feature_extraction", {}).get("status") == "success":
            newly_extracted += 1
        else:
            failed += 1
        output_records.append(out)

        if (idx + 1) % save_every == 0:
            atomic_write_jsonl(output_records, output_meta_path)
            log(f"Saved progress: {len(output_records)} / {len(records)}; new={newly_extracted}; failed={failed}")

    atomic_write_jsonl(output_records, output_meta_path)
    total_success = sum(1 for x in output_records if x.get("feature_extraction", {}).get("status") == "success")
    total_failed = sum(1 for x in output_records if x.get("feature_extraction", {}).get("status") == "failed")
    log(f"Done. Total success: {total_success}/{len(output_records)}")
    log(f"Total failed: {total_failed}/{len(output_records)}")
    log(f"Newly extracted this run: {newly_extracted}")
    log(f"Meta saved to: {output_meta_path}")
    log(f"Tensors saved in: {tensor_dir}")
    log(f"Token ids saved in: {token_dir}")
    log(f"Sidecars saved in: {sidecar_dir}")
    return output_records
