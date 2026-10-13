from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from ragprobe.prompts.judge import build_judge_prompts
from ragprobe.evaluation.answer_extraction import parse_judge_result

from ragprobe.utils.config import load_yaml
from ragprobe.utils.io import append_jsonl, read_jsonl


TRANSITIONS = ("NN", "NY", "YN", "YY")
ISSUE_TYPES = (
    "none",
    "alias",
    "granularity",
    "hypernym_subtype",
    "partial",
    "wrong_entity",
    "wrong_relation",
    "ambiguous_question",
    "bad_gold",
    "judge_too_strict",
    "judge_too_lenient",
)


def _is_unknown_answer(answer: Any) -> bool:
    text = str(answer or "").strip().lower()
    if not text:
        return True
    unknown_markers = (
        "i don" + chr(39) + "t know",
        "i do not know",
        "unknown",
        "cannot determine",
        "can" + chr(39) + "t determine",
        "not enough information",
        "insufficient information",
    )
    return any(marker in text for marker in unknown_markers)


def _transition(no_correct: bool, rag_correct: bool) -> str:
    if no_correct and rag_correct:
        return "YY"
    if no_correct and not rag_correct:
        return "YN"
    if not no_correct and rag_correct:
        return "NY"
    return "NN"


def _get_example_id(record: dict[str, Any], source_path: str, line_number: int) -> str:
    example = record.get("example") or {}
    return str(example.get("id") or record.get("example_id") or f"{source_path}:{line_number}")


def _normalize_contexts(contexts: Any, max_chars: int) -> str:
    texts: list[str] = []
    for context in contexts or []:
        if isinstance(context, dict):
            title = context.get("title")
            text = context.get("text") or context.get("paragraph_text") or ""
            if title:
                texts.append(f"{title}\n{text}")
            else:
                texts.append(str(text))
        else:
            texts.append(str(context))
    joined = "\n\n".join(texts).strip()
    if max_chars > 0 and len(joined) > max_chars:
        return joined[:max_chars] + "\n[TRUNCATED_FOR_AUDIT_PROMPT]"
    return joined


def _call_chat(client: OpenAI, model_cfg: dict[str, Any], prompt: str) -> str:
    extra_body = dict(model_cfg.get("extra_body") or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    response = client.chat.completions.create(
        model=model_cfg["name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=float(model_cfg.get("temperature", 0.0)),
        max_tokens=int(model_cfg.get("max_tokens", 512)),
        timeout=float(model_cfg.get("timeout", 120)),
        extra_body=extra_body,
    )
    return response.choices[0].message.content or ""


def _majority_vote(votes: list[bool | None], tie_break: str = "incorrect") -> bool | None:
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    yes = sum(1 for v in valid if v)
    no = len(valid) - yes
    if yes > no:
        return True
    if no > yes:
        return False
    if tie_break == "correct":
        return True
    if tie_break == "abstain":
        return None
    return False


def _vote_one_answer(client: OpenAI, model_cfg: dict[str, Any], example: dict[str, Any], answer: str, voting_cfg: dict[str, Any]) -> dict[str, Any]:
    variants = voting_cfg.get("variants") or ["strict_semantic", "alias_friendly", "decomposed_fact"]
    tie_break = str(voting_cfg.get("tie_break", "incorrect"))
    details = []
    votes = []
    for item in build_judge_prompts(example.get("answers") or [], answer, question=example.get("question", ""), variants=variants):
        raw = _call_chat(client, model_cfg, item["prompt"])
        parsed = parse_judge_result(raw)
        votes.append(parsed)
        details.append({"variant": item["name"], "raw": raw, "parsed": parsed})
    final = _majority_vote(votes, tie_break=tie_break)
    valid = [v for v in votes if v is not None]
    yes = sum(1 for v in valid if v)
    no = len(valid) - yes
    return {
        "correct": final,
        "num_correct_votes": yes,
        "num_incorrect_votes": no,
        "num_parse_failed": len(votes) - len(valid),
        "agreement": max(yes, no) / max(len(valid), 1) if valid else 0.0,
        "variants": details,
        "tie_break": tie_break,
    }


def _prompt(record: dict[str, Any], *, include_context: bool, max_context_chars: int) -> str:
    example = record.get("example") or {}
    runs = record.get("runs") or {}
    no_run = runs.get("without_context") or {}
    rag_run = runs.get("with_context") or {}
    context_text = _normalize_contexts(example.get("contexts") or [], max_context_chars)

    parts = [
        "You are auditing existing QA judge labels. Return only valid JSON, no markdown, no extra text.",
        "",
        "Decide whether each model answer is semantically correct for the question, using the gold answers as acceptable references.",
        "Strict rules:",
        "- Judge final answer correctness against the gold answer, not whether the answer is justified by the context.",
        "- Evidence/context is only for disambiguating entities or detecting bad gold. It never makes an answer correct if the final answer does not answer the question.",
        "- An answer of I do not know / unknown / cannot determine / not enough information is always incorrect unless a gold answer explicitly says unknown.",
        "- Accept aliases, stage names, abbreviations, short names vs full names, spelling variants, singular/plural, and equivalent specificity.",
        "- Do not mark gold as bad merely because the model answer is a shorter alias of the gold.",
        "- For date questions, a year-only answer is correct only if the question asks for a year or the gold/date granularity makes the year sufficient; otherwise use issue_type=granularity if debatable.",
        "- Mark bad_gold only if the gold answer itself appears inconsistent with the question or provided evidence.",
        "- If original labels are wrong only because the judge was too strict or too lenient, use judge_too_strict or judge_too_lenient unless a more specific issue type applies.",
        "",
        'Output schema exactly: {"without_context_correct": boolean, "with_context_correct": boolean, "issue_type": "none|alias|granularity|hypernym_subtype|partial|wrong_entity|wrong_relation|ambiguous_question|bad_gold|judge_too_strict|judge_too_lenient", "confidence": number, "notes": string}',
        "",
        f"Dataset: {example.get('dataset')}",
        f"Split: {example.get('split')}",
        f"Question: {example.get('question')}",
        f"Gold answers: {json.dumps(example.get('answers') or [], ensure_ascii=False)}",
    ]
    if include_context and context_text:
        parts.extend(["", "Provided evidence/context:", context_text])
    parts.extend(
        [
            "",
            f"Without-context answer: {no_run.get('extracted_answer')}",
            f"Original without-context label: {'Correct' if no_run.get('is_correct') else 'Incorrect'}",
            f"With-context answer: {rag_run.get('extracted_answer')}",
            f"Original with-context label: {'Correct' if rag_run.get('is_correct') else 'Incorrect'}",
        ]
    )
    return "\n".join(parts)


def _parse_json(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _audit_one(
    client: OpenAI,
    model_cfg: dict[str, Any],
    item: dict[str, Any],
    *,
    include_context: bool,
    max_context_chars: int,
    voting_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = item["record"]
    example = record.get("example") or {}
    runs = record.get("runs") or {}
    no_run = runs.get("without_context") or {}
    rag_run = runs.get("with_context") or {}
    original_no = bool(no_run.get("is_correct"))
    original_rag = bool(rag_run.get("is_correct"))
    original_transition = (record.get("derived") or {}).get("transition") or _transition(original_no, original_rag)

    prompt = _prompt(record, include_context=include_context, max_context_chars=max_context_chars)
    raw = ""
    vote_details = None
    try:
        voting_cfg = voting_cfg or {}
        if bool(voting_cfg.get("enabled", False)):
            no_vote = _vote_one_answer(client, model_cfg, example, no_run.get("extracted_answer"), voting_cfg)
            rag_vote = _vote_one_answer(client, model_cfg, example, rag_run.get("extracted_answer"), voting_cfg)
            audit_no = bool(no_vote["correct"]) if no_vote["correct"] is not None else original_no
            audit_rag = bool(rag_vote["correct"]) if rag_vote["correct"] is not None else original_rag
            vote_details = {"without_context": no_vote, "with_context": rag_vote}
            parsed = {"notes": "correctness decided by prompt voting"}
            raw = json.dumps(vote_details, ensure_ascii=False)
            issue_type = "none"
            confidence = min(float(no_vote.get("agreement", 0.0)), float(rag_vote.get("agreement", 0.0)))
            parse_ok = no_vote["correct"] is not None and rag_vote["correct"] is not None
            error = None if parse_ok else "all vote prompts failed for one or more answers"
        else:
            raw = _call_chat(client, model_cfg, prompt)
            parsed = _parse_json(raw)
            audit_no = bool(parsed.get("without_context_correct"))
            audit_rag = bool(parsed.get("with_context_correct"))
            issue_type = str(parsed.get("issue_type") or "none")
            if issue_type not in ISSUE_TYPES:
                issue_type = "none"
            confidence = float(parsed.get("confidence") or 0.0)
            parse_ok = True
            error = None
        if _is_unknown_answer(no_run.get("extracted_answer")):
            audit_no = False
        if _is_unknown_answer(rag_run.get("extracted_answer")):
            audit_rag = False
    except Exception as exc:
        parsed = {}
        audit_no = original_no
        audit_rag = original_rag
        issue_type = "none"
        confidence = 0.0
        parse_ok = False
        error = str(exc)

    revised_transition = _transition(audit_no, audit_rag)
    return {
        "audit_id": item["audit_id"],
        "source_path": item["source_path"],
        "line_number": item["line_number"],
        "example_id": _get_example_id(record, item["source_path"], item["line_number"]),
        "dataset": example.get("dataset"),
        "split": example.get("split"),
        "question": example.get("question"),
        "answers": example.get("answers") or [],
        "transition_original": original_transition,
        "transition_audit": revised_transition,
        "without_context_answer": no_run.get("extracted_answer"),
        "with_context_answer": rag_run.get("extracted_answer"),
        "without_context_correct_original": original_no,
        "with_context_correct_original": original_rag,
        "without_context_correct_audit": audit_no,
        "with_context_correct_audit": audit_rag,
        "label_changed": (original_no != audit_no) or (original_rag != audit_rag),
        "issue_type": issue_type,
        "confidence": confidence,
        "notes": str(parsed.get("notes") or "") if parsed else "",
        "audit_raw": raw,
        "judge_vote": vote_details,
        "parse_ok": parse_ok,
        "error": error,
    }


def _load_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    input_cfg = config.get("input", {})
    paths = input_cfg.get("generation_result_paths") or []
    datasets = set(input_cfg.get("datasets") or [])
    splits = set(input_cfg.get("split_filter") or [])
    transitions = set(input_cfg.get("transitions") or TRANSITIONS)

    candidates: list[dict[str, Any]] = []
    for path in paths:
        records = read_jsonl(path)
        for idx, record in enumerate(records, start=1):
            example = record.get("example") or {}
            runs = record.get("runs") or {}
            if "without_context" not in runs or "with_context" not in runs:
                continue
            dataset = example.get("dataset")
            split = example.get("split")
            no_correct = bool((runs.get("without_context") or {}).get("is_correct"))
            rag_correct = bool((runs.get("with_context") or {}).get("is_correct"))
            transition = (record.get("derived") or {}).get("transition") or _transition(no_correct, rag_correct)
            if datasets and dataset not in datasets:
                continue
            if splits and split not in splits:
                continue
            if transitions and transition not in transitions:
                continue
            audit_id = f"{path}:{idx}"
            candidates.append({"audit_id": audit_id, "source_path": path, "line_number": idx, "record": record})
    return candidates


def _sample(candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    sample_cfg = config.get("sampling", {})
    rng = random.Random(int(sample_cfg.get("seed", 13)))
    max_total = int(sample_cfg.get("max_total", 0) or 0)
    per_group = int(sample_cfg.get("max_per_dataset_transition", 0) or 0)
    if per_group <= 0 and max_total <= 0:
        rng.shuffle(candidates)
        return candidates

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        record = item["record"]
        example = record.get("example") or {}
        transition = (record.get("derived") or {}).get("transition")
        groups[(str(example.get("dataset")), str(transition))].append(item)

    selected: list[dict[str, Any]] = []
    for group_items in groups.values():
        rng.shuffle(group_items)
        selected.extend(group_items[:per_group] if per_group > 0 else group_items)
    rng.shuffle(selected)
    if max_total > 0:
        selected = selected[:max_total]
    return selected


def _write_summary(output_path: Path, summary_path: Path) -> None:
    records = read_jsonl(output_path)
    total = len(records)
    changed = sum(1 for r in records if r.get("label_changed"))
    parse_failed = sum(1 for r in records if not r.get("parse_ok"))
    vote_records = [r for r in records if r.get("judge_vote")]
    low_vote_agreement = 0
    for r in vote_records:
        vote = r.get("judge_vote") or {}
        agreements = [
            (vote.get("without_context") or {}).get("agreement"),
            (vote.get("with_context") or {}).get("agreement"),
        ]
        agreements = [float(x) for x in agreements if x is not None]
        if agreements and min(agreements) < 1.0:
            low_vote_agreement += 1

    by_dataset: dict[str, Counter] = defaultdict(Counter)
    by_transition: dict[str, Counter] = defaultdict(Counter)
    by_issue = Counter()
    transition_pairs = Counter()
    for record in records:
        dataset = str(record.get("dataset"))
        original = str(record.get("transition_original"))
        audited = str(record.get("transition_audit"))
        by_dataset[dataset]["total"] += 1
        by_transition[original]["total"] += 1
        if record.get("label_changed"):
            by_dataset[dataset]["changed"] += 1
            by_transition[original]["changed"] += 1
        by_issue[str(record.get("issue_type"))] += 1
        transition_pairs[f"{original}->{audited}"] += 1

    summary = {
        "num_audited": total,
        "num_label_changed": changed,
        "label_changed_rate": changed / total if total else 0.0,
        "num_parse_failed": parse_failed,
        "parse_failed_rate": parse_failed / total if total else 0.0,
        "num_vote_records": len(vote_records),
        "num_low_vote_agreement": low_vote_agreement,
        "low_vote_agreement_rate": low_vote_agreement / len(vote_records) if vote_records else 0.0,
        "by_dataset": {
            key: {**dict(value), "changed_rate": value["changed"] / value["total"] if value["total"] else 0.0}
            for key, value in sorted(by_dataset.items())
        },
        "by_original_transition": {
            key: {**dict(value), "changed_rate": value["changed"] / value["total"] if value["total"] else 0.0}
            for key, value in sorted(by_transition.items())
        },
        "issue_type_counts": dict(by_issue.most_common()),
        "transition_pair_counts": dict(transition_pairs.most_common()),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_audit(config: dict[str, Any]) -> None:
    output_cfg = config.get("output", {})
    output_path = Path(output_cfg.get("path", "outputs/audits/generation_label_audit_qwen27b.jsonl"))
    summary_path = Path(output_cfg.get("summary_path", output_path.with_suffix(".summary.json")))
    resume = bool(config.get("sampling", {}).get("resume", True))
    done = {r.get("audit_id") for r in read_jsonl(output_path)} if resume else set()

    candidates = _sample(_load_candidates(config), config)
    pending = [item for item in candidates if item["audit_id"] not in done]
    print(f"candidates={len(candidates)} done={len(done)} pending={len(pending)} output={output_path}")

    model_cfg = config.get("judge_model", {})
    client = OpenAI(
        base_url=model_cfg.get("base_url", "http://localhost:2334/v1"),
        api_key=model_cfg.get("api_key", "EMPTY"),
        timeout=float(model_cfg.get("timeout", 120)),
    )
    audit_cfg = config.get("audit", {})
    include_context = bool(audit_cfg.get("include_context", True))
    max_context_chars = int(audit_cfg.get("max_context_chars", 12000))
    voting_cfg = audit_cfg.get("voting", {})
    concurrency = int(config.get("runtime", {}).get("concurrency", 8))
    lock = threading.Lock()

    def worker(item: dict[str, Any]) -> None:
        result = _audit_one(
            client,
            model_cfg,
            item,
            include_context=include_context,
            max_context_chars=max_context_chars,
            voting_cfg=voting_cfg,
        )
        with lock:
            append_jsonl(result, output_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, item) for item in pending]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            if idx % int(config.get("runtime", {}).get("log_every", 20)) == 0:
                print(f"audited={idx}/{len(pending)}")

    _write_summary(output_path, summary_path)
    print(f"summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_audit(load_yaml(args.config))


if __name__ == "__main__":
    main()
