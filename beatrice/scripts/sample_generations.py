"""Sampling pass for Semantic Entropy / Semantic Entropy Probes (SEPs).

The stored features only contain a SINGLE (greedy) forward pass, so Semantic
Entropy — which needs K stochastic samples per question — has no inputs yet.
This script produces them: for every example in the SAME meta / fixed-scope /
labels the baselines use, it re-issues the EXACT prompt that generated the
stored answer (``runs.<run>.prompt``) to the vLLM OpenAI-compatible server with
``n=K`` at temperature>0, and records each sample's text, extracted answer and
sequence log-probability.

Output: one JSONL per split under ``--out-dir``, one row per example:
    {example_id, dataset, split, label, transition, prompt,
     greedy_answer, samples:[{text, answer, seq_logprob, num_tokens}]}

Usage (run once the Qwen server is up):
    python scripts/sample_generations.py \
        --config configs/experiment/probe/baselines/baselines_question_noctx_qwen9b.yaml \
        --setting noctx --k 10 --train-limit 4000
    python scripts/sample_generations.py \
        --config configs/experiment/probe/baselines/baselines_context_withctx_qwen9b.yaml \
        --setting withctx --k 10 --train-limit 4000
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from ragprobe.evaluation.answer_extraction import extract_final_answer
from ragprobe.features.extractor import _build_prompt as _feature_build_prompt
from ragprobe.features.extractor import _format_context_legacy
from ragprobe.prompts.qa import build_qa_prompt
from ragprobe.training.dataset import _packed_span, get_binary_label, load_feature_meta
from ragprobe.utils.config import load_yaml
from ragprobe.utils.io import write_jsonl

# --training-format: sample from the SAME prompt the feature extractor used to
# generate the training-format short answer the aligned labels grade (byte-faithful
# via extractor._build_prompt), so a consistency baseline measures the format its
# label describes. Default False keeps the eval-format behaviour bit-identical.
TRAINING_FORMAT = False
from ragprobe.utils.logging import log

RUN_KEY = {"noctx": "without_context", "withctx": "with_context"}


def _example(item: dict) -> dict:
    return item.get("example", item)


def _prompt_for(item: dict, setting: str) -> str:
    if TRAINING_FORMAT:
        ex = _example(item)
        question = str(ex.get("question") or "").strip()
        context = _format_context_legacy(item).strip() if setting == "withctx" else None
        return _feature_build_prompt(question, context)[1]
    run_key = RUN_KEY[setting]
    runs = item.get("runs", {})
    if isinstance(runs, dict) and isinstance(runs.get(run_key), dict):
        p = runs[run_key].get("prompt")
        if p:
            return p
    # Fallback: rebuild from the example (should rarely be needed).
    return build_qa_prompt(_example(item), with_context=(setting == "withctx"))


def _greedy_answer(item: dict, setting: str) -> str:
    if TRAINING_FORMAT:
        # the training-format short answer the aligned label graded, stored in the
        # feature meta's answer span (extractor's own generation)
        span = item.get("feature_extraction", {}).get("rag", {}).get("spans", {})
        at = span.get("answer_text")
        if at:
            return str(at)
    runs = item.get("runs", {})
    run = runs.get(RUN_KEY[setting], {}) if isinstance(runs, dict) else {}
    return run.get("extracted_answer", "")


def _sample_answer(text: str) -> str:
    # training format: the model answers directly (no Reasoning/Answer block), so the
    # whole stripped generation IS the answer; eval format needs the "Answer:" parse.
    return text.strip() if TRAINING_FORMAT else extract_final_answer(text)


def _sample_one(client, model_name, prompt, k, temperature, top_p, max_tokens, enable_thinking):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        n=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    samples = []
    for choice in resp.choices:
        text = choice.message.content or ""
        seq_logprob, ntok = 0.0, 0
        lp = getattr(choice, "logprobs", None)
        if lp is not None and getattr(lp, "content", None):
            for tok in lp.content:
                if tok.logprob is not None:
                    seq_logprob += float(tok.logprob)
                    ntok += 1
        samples.append({
            "text": text,
            "answer": _sample_answer(text),
            "seq_logprob": seq_logprob,
            "num_tokens": ntok,
        })
    return samples


def _process_split(items, setting, client, model_name, k, temperature, top_p,
                   max_tokens, enable_thinking, label_mode, positive_transition,
                   workers, out_path):
    def work(item):
        ex = _example(item)
        prompt = _prompt_for(item, setting)
        samples = _sample_one(client, model_name, prompt, k, temperature, top_p,
                              max_tokens, enable_thinking)
        return {
            "example_id": str(ex.get("id", "")),
            "dataset": ex.get("dataset", ""),
            "split": ex.get("split", ""),
            "label": int(get_binary_label(item, positive_transition, label_mode)),
            "transition": item.get("derived", {}).get("transition", ""),
            "prompt": prompt,
            "greedy_answer": _greedy_answer(item, setting),
            "samples": samples,
        }

    records = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, it): it for it in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=out_path.name):
            try:
                records.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                it = futures[fut]
                log(f"[sample] FAILED {str(_example(it).get('id',''))}: {type(exc).__name__}: {exc}")
    # Keep a stable order by example_id for reproducible downstream joins.
    records.sort(key=lambda r: r["example_id"])
    write_jsonl(records, str(out_path))
    log(f"[{out_path.name}] wrote {len(records)} examples (k={k})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setting", choices=["noctx", "withctx"], required=True)
    ap.add_argument("--model-config", default="configs/model/qwen3_5_9b.yaml")
    ap.add_argument("--base-url", default=None, help="Override server base_url.")
    ap.add_argument("--model-name", default=None, help="Override served model name.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--train-limit", type=int, default=4000,
                    help="Subsample train (for SEP labels); 0 = skip train.")
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "eval"],
                    help="Which split groups to sample: train / validation / eval.")
    ap.add_argument("--eval-names", nargs="*", default=None,
                    help="If set, only sample these eval_meta_sets (by name) when 'eval' is in --splits.")
    ap.add_argument("--meta-paths", nargs="*", default=None,
                    help="Override: sample these meta paths (no partition) as a single group.")
    ap.add_argument("--group-name", default=None, help="Output group name for --meta-paths.")
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test cap per split.")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--training-format", action="store_true",
                    help="Sample from the feature-extractor (training-format) prompt so a "
                         "consistency baseline matches the aligned short-answer labels.")
    args = ap.parse_args()
    global TRAINING_FORMAT
    TRAINING_FORMAT = bool(args.training_format)

    cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    data_cfg = cfg["data"]
    probe_cfg = cfg.get("probe", {})
    label_mode = probe_cfg.get("label_mode", "without_context_correct")
    positive_transition = probe_cfg.get("positive_transition", "NY")

    base_url = args.base_url or model_cfg.get("base_url")
    model_name = args.model_name or model_cfg.get("name")
    enable_thinking = bool(model_cfg.get("enable_thinking", False))
    client = OpenAI(api_key=model_cfg.get("api_key", "EMPTY"), base_url=base_url,
                    timeout=model_cfg.get("timeout", 120), max_retries=3)
    log(f"server={base_url} model={model_name} k={args.k} T={args.temperature} setting={args.setting}")

    out_dir = Path(args.out_dir or f"data/se_samples/{args.setting}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    def run_group(name, items):
        if args.limit:
            items = items[:args.limit]
        log(f"sampling group={name} n={len(items)}")
        _process_split(items, args.setting, client, model_name, args.k, args.temperature,
                       args.top_p, args.max_tokens, enable_thinking, label_mode,
                       positive_transition, args.workers, out_dir / f"{name}.jsonl")

    # Override: sample an explicit meta (e.g. matched rewritten-context records) as one group.
    if args.meta_paths:
        meta = load_feature_meta(args.meta_paths, splits=None, partition=None)
        run_group(args.group_name or "custom", meta)
        return

    if "train" in args.splits and args.train_limit:
        train_meta = load_feature_meta(
            data_cfg["train_meta_paths"], splits=data_cfg.get("train_splits"),
            partition=data_cfg.get("train_partition"))
        rng.shuffle(train_meta)
        run_group("train", train_meta[:args.train_limit])

    if "validation" in args.splits:
        val_meta = load_feature_meta(
            data_cfg["validation_meta_paths"], splits=data_cfg.get("validation_splits"),
            partition=data_cfg.get("validation_partition"))
        run_group("validation", val_meta)

    if "eval" in args.splits:
        for extra in data_cfg.get("eval_meta_sets", []) or []:
            if args.eval_names and extra["name"] not in args.eval_names:
                continue
            meta = load_feature_meta(extra["meta_paths"], splits=extra.get("splits"),
                                     partition=extra.get("partition"))
            run_group(extra["name"], meta)


if __name__ == "__main__":
    main()
