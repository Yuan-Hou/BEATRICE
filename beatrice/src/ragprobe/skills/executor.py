from __future__ import annotations

from typing import Any

from openai import OpenAI

from ragprobe.skills import prompts


class SkillExecutor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.generation_cfg = config.get("generation_model", {})
        self.judge_cfg = config.get("judge_model", self.generation_cfg)
        self.generation_client = self._client(self.generation_cfg)
        self.judge_client = self._client(self.judge_cfg)

    @staticmethod
    def _client(cfg: dict[str, Any]) -> OpenAI:
        return OpenAI(
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key", "EMPTY"),
            timeout=float(cfg.get("timeout", 120)),
        )

    @staticmethod
    def _clean_answer(text: str) -> str:
        text = (text or "").strip()
        if text.lower().startswith("answer:"):
            return text.split(":", 1)[1].strip()
        return text

    def _chat(self, client: OpenAI, cfg: dict[str, Any], prompt: str) -> dict[str, Any]:
        try:
            extra_body = dict(cfg.get("extra_body") or {})
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            response = client.chat.completions.create(
                model=cfg["name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(cfg.get("max_tokens", 256)),
                temperature=float(cfg.get("temperature", 0.0)),
                timeout=float(cfg.get("timeout", 120)),
                extra_body=extra_body,
            )
            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            usage_dict = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            if usage_dict["total_tokens"] <= 0:
                prompt_estimate = max(len(prompt) // 4, 1)
                completion_estimate = max(len(text) // 4, 0)
                usage_dict = {
                    "prompt_tokens": prompt_estimate,
                    "completion_tokens": completion_estimate,
                    "total_tokens": prompt_estimate + completion_estimate,
                    "estimated": True,
                }
            if not text.strip():
                return {"ok": False, "output": "", "error": "empty_model_output", "usage": usage_dict}
            return {"ok": True, "output": text.strip(), "error": None, "usage": usage_dict}
        except Exception as exc:
            return {"ok": False, "output": "", "error": str(exc), "usage": {"prompt_tokens": max(len(prompt) // 4, 1), "completion_tokens": 0, "total_tokens": max(len(prompt) // 4, 1), "estimated": True}}

    def answer_with_rag(self, question: str, contexts) -> dict[str, Any]:
        result = self._chat(
            self.generation_client,
            self.generation_cfg,
            prompts.answer_with_rag_prompt(question, contexts),
        )
        if result["ok"]:
            result["output"] = self._clean_answer(result["output"])
        return result

    def answer_no_rag(self, question: str) -> dict[str, Any]:
        result = self._chat(
            self.generation_client,
            self.generation_cfg,
            prompts.answer_no_rag_prompt(question),
        )
        if result["ok"]:
            result["output"] = self._clean_answer(result["output"])
        return result

    def evidence_focus(self, question: str, contexts) -> dict[str, Any]:
        return self._chat(
            self.generation_client,
            self.generation_cfg,
            prompts.evidence_focus_prompt(question, contexts),
        )

    def decompose_then_focus(self, question: str, contexts) -> dict[str, Any]:
        return self._chat(
            self.generation_client,
            self.generation_cfg,
            prompts.decompose_then_focus_prompt(question, contexts),
        )

    def rewrite_context(self, question: str, contexts) -> dict[str, Any]:
        return self._chat(
            self.generation_client,
            self.generation_cfg,
            prompts.rewrite_context_prompt(question, contexts),
        )


    def decompose_question_json(self, question: str, *, max_subquestions: int = 3) -> dict[str, Any]:
        return self._chat(
            self.generation_client,
            self.generation_cfg,
            prompts.decompose_question_json_prompt(question, max_subquestions=max_subquestions),
        )

    def decompose_question_with_context_json(self, question: str, contexts, *, max_subquestions: int = 3) -> dict[str, Any]:
        return self._chat(
            self.generation_client,
            self.generation_cfg,
            prompts.decompose_question_with_context_json_prompt(question, contexts, max_subquestions=max_subquestions),
        )

    def rewrite_child_question(self, parent_question: str, prior_answers, child_question: str) -> dict[str, Any]:
        result = self._chat(
            self.generation_client,
            self.generation_cfg,
            prompts.rewrite_child_question_prompt(parent_question, prior_answers, child_question),
        )
        if result.get("ok") and result.get("output", "").lower().startswith("question:"):
            result["output"] = result["output"].split(":", 1)[1].strip()
        return result

    def synthesize_from_subanswers(self, question: str, subanswers, contexts=None) -> dict[str, Any]:
        result = self._chat(
            self.generation_client,
            self.generation_cfg,
            prompts.synthesize_from_subanswers_prompt(question, subanswers, contexts),
        )
        if result.get("ok"):
            result["output"] = self._clean_answer(result["output"])
        return result

    def judge_answer(self, question: str, answers, model_answer: str) -> dict[str, Any]:
        result = self._chat(
            self.judge_client,
            self.judge_cfg,
            prompts.judge_prompt(question, answers, model_answer),
        )
        if not result["ok"]:
            result["correct"] = False
            return result
        verdict = result["output"].strip().lower()
        result["correct"] = verdict.startswith("correct")
        return result


def split_context_output(text: str) -> list[str]:
    contexts = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            stripped = stripped[1:].strip()
        if stripped.lower().startswith(("focused evidence:", "compressed context:", "subquestions and evidence:", "rewritten query:")):
            continue
        contexts.append(stripped)
    return contexts or [text.strip()] if text.strip() else []
