from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional

from ragprobe.evaluation.metrics import transition_from_correctness
from ragprobe.triage.policy import (
    NO_RAG,
    SKILL_ROUTE,
    UNCERTAIN,
    USE_RAG,
    WITH_CONTEXT,
    WITHOUT_CONTEXT,
    ProbeTriagePolicy,
)
from ragprobe.utils.io import read_jsonl, write_jsonl
from ragprobe.utils.logging import log


def _example(record: dict[str, Any]) -> dict[str, Any]:
    if isinstance(record.get("example"), dict):
        return record["example"]
    return {
        "id": str(record.get("id") or record.get("example_id") or record.get("question") or ""),
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "question": record.get("question", ""),
        "answers": record.get("answers", []),
        "contexts": record.get("contexts", record.get("context", [])),
    }


def _example_id(record: dict[str, Any]) -> str:
    ex = _example(record)
    return str(ex.get("id") or ex.get("question") or "")


def _question(record: dict[str, Any]) -> str:
    return str(_example(record).get("question") or record.get("question") or "")


def _split(record: dict[str, Any]) -> Optional[str]:
    return _example(record).get("split") or record.get("split")


def _transition(record: dict[str, Any]) -> Optional[str]:
    transition = record.get("derived", {}).get("transition")
    if transition:
        return transition

    if "transition" in record:
        return record["transition"]

    without_correct = _run_is_correct(record, WITHOUT_CONTEXT)
    with_correct = _run_is_correct(record, WITH_CONTEXT)
    if without_correct is not None and with_correct is not None:
        return transition_from_correctness(bool(without_correct), bool(with_correct))

    if "is_correct_without_context" in record or "is_correct_with_context" in record:
        return transition_from_correctness(
            bool(record.get("is_correct_without_context", False)),
            bool(record.get("is_correct_with_context", False)),
        )

    return None


def _run(record: dict[str, Any], run_name: str) -> dict[str, Any]:
    return record.get("runs", {}).get(run_name, {})


def _run_answer(record: dict[str, Any], run_name: str) -> Any:
    run = _run(record, run_name)
    return run.get("extracted_answer", run.get("answer"))


def _run_is_correct(record: dict[str, Any], run_name: str) -> Optional[bool]:
    run = _run(record, run_name)
    if "is_correct" in run:
        return bool(run["is_correct"])

    key = "is_correct_with_context" if run_name == WITH_CONTEXT else "is_correct_without_context"
    if key in record:
        return bool(record[key])
    return None


def _context_chars(record: dict[str, Any]) -> int:
    contexts = _example(record).get("contexts", [])
    if isinstance(contexts, str):
        return len(contexts)
    if not isinstance(contexts, list):
        return 0

    total = 0
    for ctx in contexts:
        if isinstance(ctx, dict):
            total += len(str(ctx.get("text", "")))
        else:
            total += len(str(ctx))
    return total


def _filter_generation_records(records: Iterable[dict[str, Any]], split_filter=None) -> list[dict[str, Any]]:
    if not split_filter:
        return list(records)

    splits = set(split_filter)
    return [record for record in records if _split(record) in splits]


def _index_generation_records(records: list[dict[str, Any]]):
    by_id = {}
    by_question = {}
    for idx, record in enumerate(records):
        ex_id = _example_id(record)
        question = _question(record)
        if ex_id:
            by_id[ex_id] = record
        if question:
            by_question[question] = record
    return by_id, by_question


def _join_records(
    generation_records: list[dict[str, Any]],
    probe_records: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    by_id, by_question = _index_generation_records(generation_records)
    joined = []
    missing = 0
    joined_by = Counter()

    for pred in probe_records:
        gen = None
        ex_id = pred.get("example_id") or pred.get("id")
        if ex_id:
            gen = by_id.get(str(ex_id))
            if gen is not None:
                joined_by["example_id"] += 1

        if gen is None and pred.get("question"):
            gen = by_question.get(str(pred["question"]))
            if gen is not None:
                joined_by["question"] += 1

        if gen is None and pred.get("idx") is not None:
            idx = int(pred["idx"])
            if 0 <= idx < len(generation_records):
                gen = generation_records[idx]
                joined_by["idx"] += 1

        if gen is None:
            missing += 1
            continue

        joined.append((gen, pred))

    return joined, {
        "probe_records": len(probe_records),
        "generation_records": len(generation_records),
        "joined": len(joined),
        "missing": missing,
        "joined_by_example_id": joined_by.get("example_id", 0),
        "joined_by_question": joined_by.get("question", 0),
        "joined_by_idx": joined_by.get("idx", 0),
    }


def _label_helpful(pred: dict[str, Any], transition: Optional[str]) -> int:
    if "label" in pred:
        return int(pred["label"])
    return 1 if transition == "NY" else 0


def _regression_fields(pred: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in ["regression_pred", "regression_confidence", "regression_probs"]:
        if key in pred:
            out[key] = pred[key]

    if "regression_pred" not in out and "rag_signal_prediction_name" in pred:
        out["regression_pred"] = pred["rag_signal_prediction_name"]
    if "regression_confidence" not in out and "rag_signal_probabilities" in pred:
        pred_name = pred.get("rag_signal_prediction_name")
        probs = pred.get("rag_signal_probabilities") or {}
        if pred_name in probs:
            out["regression_confidence"] = probs[pred_name]
    return out


def _run_triage(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    policy_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = ProbeTriagePolicy(policy_cfg)
    records = []

    for gen, pred in joined:
        ex = _example(gen)
        transition = _transition(gen) or pred.get("transition")
        p_helpful = float(pred.get("helpful_probability", pred.get("p_helpful", 0.0)))
        decision = policy.decide(pred)
        chosen_run = decision["chosen_run"]

        record = {
            "example_id": ex.get("id") or pred.get("example_id") or _question(gen),
            "dataset": ex.get("dataset") or pred.get("dataset"),
            "split": ex.get("split") or pred.get("split"),
            "question": ex.get("question") or pred.get("question"),
            "answers": ex.get("answers") or pred.get("answers", []),
            "transition": transition,
            "p_helpful": p_helpful,
            "label_helpful": _label_helpful(pred, transition),
            "policy_action": decision["action"],
            "chosen_run": chosen_run,
            "chosen_answer": _run_answer(gen, chosen_run),
            "chosen_is_correct": _run_is_correct(gen, chosen_run),
            "without_context_is_correct": _run_is_correct(gen, WITHOUT_CONTEXT),
            "with_context_is_correct": _run_is_correct(gen, WITH_CONTEXT),
            "recommended_skill": decision.get("recommended_skill"),
            "reason": decision.get("reason"),
        }
        record.update(_regression_fields(pred))
        record["_context_chars"] = _context_chars(gen)
        records.append(record)

    return records, _compute_metrics(records)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _accuracy(correct_values: Iterable[Optional[bool]]) -> float:
    values = [bool(v) for v in correct_values if v is not None]
    return _rate(sum(values), len(values))


def _compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    num_examples = len(records)
    transition_counts = Counter(record.get("transition") or "UNKNOWN" for record in records)
    action_counts = Counter(record["policy_action"] for record in records)
    action_by_transition = defaultdict(Counter)

    for record in records:
        action_by_transition[record.get("transition") or "UNKNOWN"][record["policy_action"]] += 1

    no_rag_accuracy = _accuracy(record["without_context_is_correct"] for record in records)
    always_rag_accuracy = _accuracy(record["with_context_is_correct"] for record in records)
    triage_accuracy = _accuracy(record["chosen_is_correct"] for record in records)

    ny_records = [record for record in records if record.get("transition") == "NY"]
    yn_records = [record for record in records if record.get("transition") == "YN"]
    with_context_count = sum(1 for record in records if record["chosen_run"] == WITH_CONTEXT)
    without_context_count = sum(1 for record in records if record["chosen_run"] == WITHOUT_CONTEXT)

    chosen_context_chars = [
        int(record.get("_context_chars", 0))
        for record in records
        if record["chosen_run"] == WITH_CONTEXT
    ]
    total_context_chars = sum(chosen_context_chars)
    transition_names = ["NN", "NY", "YN", "YY", "UNKNOWN"]
    action_names = [USE_RAG, NO_RAG, UNCERTAIN, SKILL_ROUTE]

    metrics = {
        "num_examples": num_examples,
        "no_rag_accuracy": no_rag_accuracy,
        "always_rag_accuracy": always_rag_accuracy,
        "triage_accuracy": triage_accuracy,
        "triage_delta_vs_no_rag": triage_accuracy - no_rag_accuracy,
        "triage_delta_vs_always_rag": triage_accuracy - always_rag_accuracy,
        "transition_counts": {
            name: transition_counts.get(name, 0)
            for name in transition_names
            if name != "UNKNOWN" or transition_counts.get(name, 0) > 0
        },
        "action_counts": {
            name: action_counts.get(name, 0)
            for name in action_names
        },
        "action_by_transition": {
            transition: {
                action: action_by_transition[transition].get(action, 0)
                for action in action_names
            }
            for transition in transition_names
            if transition != "UNKNOWN"
            or action_by_transition.get(transition)
        },
        "ny_retention_rate": _rate(
            sum(1 for record in ny_records if record["chosen_run"] == WITH_CONTEXT),
            len(ny_records),
        ),
        "yn_avoidance_rate": _rate(
            sum(1 for record in yn_records if record["chosen_run"] == WITHOUT_CONTEXT),
            len(yn_records),
        ),
        "rag_usage_rate": _rate(with_context_count, num_examples),
        "no_rag_usage_rate": _rate(without_context_count, num_examples),
        "avg_context_chars_when_chosen": _rate(total_context_chars, with_context_count),
        "approx_token_cost": total_context_chars / 4,
    }
    return metrics


def _objective(metrics: dict[str, Any], objective: str, sweep_cfg: dict[str, Any]) -> float:
    if objective == "triage_accuracy":
        return float(metrics["triage_accuracy"])
    if objective == "accuracy_minus_rag_usage_penalty":
        penalty = float(sweep_cfg.get("rag_usage_penalty", 0.05))
        return float(metrics["triage_accuracy"]) - penalty * float(metrics["rag_usage_rate"])
    if objective == "control_score":
        yn_weight = float(sweep_cfg.get("yn_avoidance_weight", 0.05))
        ny_loss_weight = float(sweep_cfg.get("ny_retention_loss_weight", 0.02))
        rag_penalty = float(sweep_cfg.get("rag_usage_penalty", 0.0))
        return (
            float(metrics["triage_accuracy"])
            + yn_weight * float(metrics["yn_avoidance_rate"])
            - ny_loss_weight * (1.0 - float(metrics["ny_retention_rate"]))
            - rag_penalty * float(metrics["rag_usage_rate"])
        )
    raise ValueError(f"Unknown sweep objective: {objective}")


def _passes_sweep_constraints(metrics: dict[str, Any], sweep_cfg: dict[str, Any]) -> bool:
    constraints = {
        "min_triage_accuracy": metrics["triage_accuracy"] >= float("-inf"),
        "min_ny_retention_rate": metrics["ny_retention_rate"] >= 0.0,
        "min_yn_avoidance_rate": metrics["yn_avoidance_rate"] >= 0.0,
        "max_rag_usage_rate": metrics["rag_usage_rate"] <= 1.0,
    }
    if "min_triage_accuracy" in sweep_cfg:
        constraints["min_triage_accuracy"] = (
            metrics["triage_accuracy"] >= float(sweep_cfg["min_triage_accuracy"])
        )
    if "min_ny_retention_rate" in sweep_cfg:
        constraints["min_ny_retention_rate"] = (
            metrics["ny_retention_rate"] >= float(sweep_cfg["min_ny_retention_rate"])
        )
    if "min_yn_avoidance_rate" in sweep_cfg:
        constraints["min_yn_avoidance_rate"] = (
            metrics["yn_avoidance_rate"] >= float(sweep_cfg["min_yn_avoidance_rate"])
        )
    if "max_rag_usage_rate" in sweep_cfg:
        constraints["max_rag_usage_rate"] = (
            metrics["rag_usage_rate"] <= float(sweep_cfg["max_rag_usage_rate"])
        )
    return all(constraints.values())


def _sweep(joined, base_policy_cfg: dict[str, Any], sweep_cfg: dict[str, Any]) -> dict[str, Any]:
    objective = sweep_cfg.get("objective", "triage_accuracy")
    best = None
    results = []
    skipped_by_constraints = 0

    for low in sweep_cfg.get("low_values", []):
        for high in sweep_cfg.get("high_values", []):
            if float(low) >= float(high):
                continue
            for uncertain_action in sweep_cfg.get("uncertain_actions", ["use_rag"]):
                policy_cfg = deepcopy(base_policy_cfg)
                policy_cfg["name"] = policy_cfg.get("name", "three_zone")
                policy_cfg["low_threshold"] = float(low)
                policy_cfg["high_threshold"] = float(high)
                policy_cfg["uncertain_action"] = uncertain_action

                _, metrics = _run_triage(joined, policy_cfg)
                if not _passes_sweep_constraints(metrics, sweep_cfg):
                    skipped_by_constraints += 1
                    continue

                score = _objective(metrics, objective, sweep_cfg)
                item = {
                    "policy": policy_cfg,
                    "objective": objective,
                    "objective_value": score,
                    "metrics": metrics,
                }
                results.append(item)
                if best is None or score > best["objective_value"]:
                    best = item

    if best is None:
        raise RuntimeError("Sweep produced no valid policies. Check low/high threshold grids.")

    return {
        "best_policy": best["policy"],
        "best_objective": objective,
        "best_objective_value": best["objective_value"],
        "num_sweep_trials": len(results),
        "num_sweep_trials_skipped_by_constraints": skipped_by_constraints,
        "sweep_results": results,
    }


def _strip_internal_fields(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_records = []
    for record in records:
        out = dict(record)
        out.pop("_context_chars", None)
        clean_records.append(out)
    return clean_records


def run_probe_triage(config: dict[str, Any]):
    input_cfg = config["input"]
    output_cfg = config["output"]
    policy_cfg = deepcopy(config.get("policy", {}))
    sweep_cfg = config.get("sweep", {})

    generation_records = read_jsonl(input_cfg["generation_result_path"])
    generation_records = _filter_generation_records(
        generation_records,
        split_filter=input_cfg.get("split_filter"),
    )
    probe_records = read_jsonl(input_cfg["probe_prediction_path"])

    joined, join_stats = _join_records(generation_records, probe_records)
    log(
        "Probe triage join: "
        f"generation={join_stats['generation_records']}, probe={join_stats['probe_records']}, "
        f"joined={join_stats['joined']}, missing={join_stats['missing']}, "
        f"by_id={join_stats['joined_by_example_id']}, "
        f"by_question={join_stats['joined_by_question']}, by_idx={join_stats['joined_by_idx']}"
    )
    if not joined:
        raise RuntimeError("No joined probe/generation records. Check input paths and split_filter.")

    sweep_summary = None
    if sweep_cfg.get("enabled", False):
        sweep_summary = _sweep(joined, policy_cfg, sweep_cfg)
        policy_cfg = sweep_summary["best_policy"]
        log(
            "Probe triage sweep best: "
            f"objective={sweep_summary['best_objective']}, "
            f"value={sweep_summary['best_objective_value']:.4f}, "
            f"policy={policy_cfg}"
        )

    triage_records, metrics = _run_triage(joined, policy_cfg)
    metrics["policy"] = policy_cfg
    metrics["join_stats"] = join_stats
    if sweep_summary is not None:
        metrics["best_policy"] = sweep_summary["best_policy"]
        metrics["best_objective"] = sweep_summary["best_objective"]
        metrics["best_objective_value"] = sweep_summary["best_objective_value"]
        metrics["num_sweep_trials"] = sweep_summary["num_sweep_trials"]
        metrics["num_sweep_trials_skipped_by_constraints"] = sweep_summary[
            "num_sweep_trials_skipped_by_constraints"
        ]
        metrics["sweep_results"] = sweep_summary["sweep_results"]

    write_jsonl(_strip_internal_fields(triage_records), output_cfg["triage_result_path"])

    metrics_path = Path(output_cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    log(f"Probe triage results saved to {output_cfg['triage_result_path']}")
    log(f"Probe triage metrics saved to {metrics_path}")
    log(
        "Probe triage metrics: "
        f"no_rag={metrics['no_rag_accuracy']:.4f}, "
        f"always_rag={metrics['always_rag_accuracy']:.4f}, "
        f"triage={metrics['triage_accuracy']:.4f}, "
        f"ny_retention={metrics['ny_retention_rate']:.4f}, "
        f"yn_avoidance={metrics['yn_avoidance_rate']:.4f}, "
        f"rag_usage={metrics['rag_usage_rate']:.4f}"
    )
    return triage_records, metrics
