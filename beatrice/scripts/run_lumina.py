"""LUMINA (ICLR'26, Yeh et al., 2509.21875) baseline via the official class.

Uses third_party/LUMINA/lumina.py UNMODIFIED. LUMINA scores a RAG answer by
  hallucination = lambda * IPR - (1 - lambda) * MMD
where MMD (external-context utilization) compares the answer-token distributions
under the real vs a random retrieved context (two forward passes), and IPR
(internal-knowledge utilization) is a logit-lens trajectory score across layers.
The official class already special-cases the `model.model.language_model` wrapper,
i.e. our AutoModelForImageTextToText Qwen, so it is used as-is.

Per example we feed:
  prompt_w_context        = our with-context user prompt (_build_prompt)
  prompt_w_random_context = same question, context swapped for a random OTHER
                            example's context (the paper's "random documents")
  response                = the stored greedy answer (teacher-forced, 1:1 with feats)
The official predict() returns per-answer-token mmd/ipr vectors; we mean-pool to a
scalar per example. lambda is tuned on validation AUROC; the final stored score is
oriented so higher = more likely CORRECT (consistent with the other baselines).

Run (vllm2 env; avoid GPUs 0,1):
    CUDA_VISIBLE_DEVICES=4 .../vllm2/bin/python scripts/run_lumina.py \
        --config configs/experiment/probe/baselines/baselines_context_withctx_qwen9b.yaml \
        --group validation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "LUMINA"))
from lumina import LUMINA  # noqa: E402

from ragprobe.features.extractor import _format_context_legacy, _get_example  # noqa: E402
from ragprobe.prompts.qa import LEGACY_MUSIQUE_FEATURE_PROMPT  # noqa: E402
from ragprobe.training.dataset import get_binary_label, load_feature_meta  # noqa: E402
from ragprobe.utils.config import load_yaml  # noqa: E402
from ragprobe.utils.logging import log  # noqa: E402

GROUPS = ["validation", "test_id", "popqa_ood"]


def _stored_answer_text(record: dict) -> str | None:
    try:
        sc_path = record["feature_extraction"]["rag"].get("sidecar_path")
    except Exception:
        sc_path = None
    if sc_path and Path(sc_path).exists():
        try:
            sc = torch.load(sc_path, map_location="cpu", weights_only=False)
            if sc.get("answer_text"):
                return str(sc["answer_text"])
        except Exception:
            pass
    runs = record.get("runs", {})
    for key in ("with_context", "without_context"):
        r = runs.get(key) if isinstance(runs, dict) else None
        if isinstance(r, dict) and r.get("extracted_answer"):
            return str(r["extracted_answer"])
    return None


def _load_group(data_cfg, group):
    if group == "validation":
        return load_feature_meta(data_cfg["validation_meta_paths"], splits=data_cfg.get("validation_splits"),
                                 partition=data_cfg.get("validation_partition"))
    extra = next(e for e in data_cfg["eval_meta_sets"] if e["name"] == group)
    return load_feature_meta(extra["meta_paths"], splits=extra.get("splits"), partition=extra.get("partition"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--group", required=True, help="validation | test_id | popqa_ood")
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/lumina_features/withctx")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    probe_cfg = cfg.get("probe", {})
    label_mode = probe_cfg.get("label_mode", "with_context_correct")
    positive_transition = probe_cfg.get("positive_transition", "NY")

    meta = _load_group(data_cfg, args.group)
    meta.sort(key=lambda m: str(_get_example(m).get("id", "")))
    if args.limit:
        meta = meta[:args.limit]
    # random-context pool: every example's own context, shuffled with a fixed seed so
    # each example is paired with a different example's documents.
    contexts = [_format_context_legacy(r).strip() for r in meta]
    rng = random.Random(args.seed)
    perm = list(range(len(meta)))
    rng.shuffle(perm)
    # ensure no example keeps its own context as the "random" one
    for i in range(len(perm)):
        if perm[i] == i:
            perm[i] = perm[(i + 1) % len(perm)]
    log(f"group={args.group} -> {len(meta)} records")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    log(f"loading {args.model_name} ...")
    # fp32: the official lumina.py forces top-k probs to float() but leaves the cosine
    # kernel in the model dtype, so bf16/fp16 hits a matmul dtype mismatch. Running the
    # model in fp32 makes every tensor match WITHOUT editing their code (stays faithful).
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, dtype=torch.float32).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    detector = LUMINA(model, processor.tokenizer, kernel="cosine", lam=0.5, device="cuda")

    rows, index, skipped = [], [], 0
    for i, record in enumerate(tqdm(meta, desc=f"lumina/{args.group}")):
        ex = _get_example(record)
        question = str(ex.get("question") or "").strip()
        ctx = contexts[i]
        rand_ctx = contexts[perm[i]]
        answer_text = _stored_answer_text(record)
        if not question or not ctx or not rand_ctx or not answer_text:
            skipped += 1
            continue
        prompt = LEGACY_MUSIQUE_FEATURE_PROMPT.substitute(context=ctx, question=question)
        prompt_rand = LEGACY_MUSIQUE_FEATURE_PROMPT.substitute(context=rand_ctx, question=question)
        try:
            with torch.no_grad():
                hscore, mmd, ipr = detector.predict(prompt, prompt_rand, answer_text)
            rows.append((float(mmd.mean().item()), float(ipr.mean().item())))
            index.append({
                "example_id": str(ex.get("id", "")), "dataset": ex.get("dataset", ""),
                "split": ex.get("split", ""),
                "label": int(get_binary_label(record, positive_transition, label_mode)),
            })
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            continue

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 2), np.float32)
    np.savez(out_dir / f"{args.group}.npz", mmd=arr[:, 0], ipr=arr[:, 1])
    (out_dir / f"{args.group}.index.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in index) + "\n", encoding="utf-8")
    log(f"wrote {len(index)} examples (skipped {skipped}) -> {out_dir}/{args.group}.npz")


if __name__ == "__main__":
    main()
