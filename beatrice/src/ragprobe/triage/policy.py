from __future__ import annotations

from typing import Any, Optional


USE_RAG = "USE_RAG"
NO_RAG = "NO_RAG"
UNCERTAIN = "UNCERTAIN"
SKILL_ROUTE = "SKILL_ROUTE"

WITH_CONTEXT = "with_context"
WITHOUT_CONTEXT = "without_context"

REGRESSION_SKILLS = {
    "Attention Bias": "evidence_focus",
    "Logic Drift": "decompose_question",
    "Hierarchy Misalignment": "granularity_check",
    "Over-correction": "loosen_faithfulness_or_use_parametric",
    "Others": "rewrite_query",
}


def _action_to_run(action: str) -> str:
    if action == "use_rag":
        return WITH_CONTEXT
    if action == "no_rag":
        return WITHOUT_CONTEXT
    raise ValueError(f"Unknown fallback action: {action}")


class ProbeTriagePolicy:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.name = config.get("name", "three_zone")
        self.threshold = float(config.get("threshold", 0.5))
        self.high_threshold = float(config.get("high_threshold", 0.75))
        self.low_threshold = float(config.get("low_threshold", 0.35))
        self.uncertain_action = config.get("uncertain_action", "use_rag")
        self.use_regression_head = bool(config.get("use_regression_head", False))
        self.regression_threshold = float(config.get("regression_threshold", 0.6))
        self.skill_fallback_action = config.get("skill_fallback_action", "no_rag")

        if self.name not in {"hard_gate", "three_zone", "regression_aware_three_zone"}:
            raise ValueError(f"Unknown triage policy: {self.name}")

    def decide(self, prediction: dict[str, Any]) -> dict[str, Any]:
        p_helpful = float(prediction.get("helpful_probability", prediction.get("p_helpful", 0.0)))

        if self.name == "hard_gate":
            return self._hard_gate(p_helpful)

        decision = self._three_zone(p_helpful)
        if self.name == "regression_aware_three_zone":
            return self._apply_regression_awareness(decision, prediction, p_helpful)
        return decision

    def _hard_gate(self, p_helpful: float) -> dict[str, Any]:
        if p_helpful >= self.threshold:
            return {
                "action": USE_RAG,
                "chosen_run": WITH_CONTEXT,
                "reason": f"p_helpful={p_helpful:.4f} >= threshold={self.threshold:.4f}",
                "recommended_skill": None,
            }

        return {
            "action": NO_RAG,
            "chosen_run": WITHOUT_CONTEXT,
            "reason": f"p_helpful={p_helpful:.4f} < threshold={self.threshold:.4f}",
            "recommended_skill": None,
        }

    def _three_zone(self, p_helpful: float) -> dict[str, Any]:
        if p_helpful >= self.high_threshold:
            return {
                "action": USE_RAG,
                "chosen_run": WITH_CONTEXT,
                "reason": f"p_helpful={p_helpful:.4f} >= high_threshold={self.high_threshold:.4f}",
                "recommended_skill": None,
            }

        if p_helpful <= self.low_threshold:
            return {
                "action": NO_RAG,
                "chosen_run": WITHOUT_CONTEXT,
                "reason": f"p_helpful={p_helpful:.4f} <= low_threshold={self.low_threshold:.4f}",
                "recommended_skill": None,
            }

        chosen_run = _action_to_run(self.uncertain_action)
        return {
            "action": UNCERTAIN,
            "chosen_run": chosen_run,
            "reason": (
                f"low_threshold={self.low_threshold:.4f} < p_helpful={p_helpful:.4f} "
                f"< high_threshold={self.high_threshold:.4f}; uncertain_action={self.uncertain_action}"
            ),
            "recommended_skill": None,
        }

    def _apply_regression_awareness(
        self,
        decision: dict[str, Any],
        prediction: dict[str, Any],
        p_helpful: float,
    ) -> dict[str, Any]:
        if not self.use_regression_head:
            return decision

        regression_pred = self._get_regression_pred(prediction)
        regression_confidence = self._get_regression_confidence(prediction)
        if regression_pred is None or regression_confidence is None:
            return decision

        if p_helpful >= self.high_threshold:
            return decision

        if regression_confidence < self.regression_threshold:
            return decision

        recommended_skill = REGRESSION_SKILLS.get(regression_pred, REGRESSION_SKILLS["Others"])
        chosen_run = _action_to_run(self.skill_fallback_action)
        return {
            "action": SKILL_ROUTE,
            "chosen_run": chosen_run,
            "reason": (
                f"regression_pred={regression_pred}, confidence={regression_confidence:.4f} "
                f">= regression_threshold={self.regression_threshold:.4f}; "
                f"skill_fallback_action={self.skill_fallback_action}"
            ),
            "recommended_skill": recommended_skill,
        }

    @staticmethod
    def _get_regression_pred(prediction: dict[str, Any]) -> Optional[str]:
        pred = prediction.get("regression_pred")
        if pred:
            return str(pred)

        pred = prediction.get("rag_signal_prediction_name")
        if pred and pred != "Not Regression":
            return str(pred)

        return None

    @staticmethod
    def _get_regression_confidence(prediction: dict[str, Any]) -> Optional[float]:
        if prediction.get("regression_confidence") is not None:
            return float(prediction["regression_confidence"])

        probs = prediction.get("regression_probs")
        if isinstance(probs, dict) and probs:
            return float(max(probs.values()))
        if isinstance(probs, list) and probs:
            return float(max(probs))

        probs = prediction.get("rag_signal_probabilities")
        pred = prediction.get("rag_signal_prediction_name")
        if isinstance(probs, dict) and pred in probs:
            return float(probs[pred])

        return None
