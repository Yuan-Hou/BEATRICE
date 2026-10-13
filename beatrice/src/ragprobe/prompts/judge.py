from __future__ import annotations

from string import Template
from typing import Any


LIST_ANSWER_JUDGE = Template("""Given the list of acceptable correct answers: $correct_answers
And the model's answer: $model_answer
Determine if the model's answer is correct. It only needs to logically match ONE of the acceptable answers.
Respond with "Correct" if the model's answer matches one acceptable correct answer, otherwise respond with "Incorrect". No additional explanation is needed, only respond with one of the two words.""")

STRICT_SEMANTIC_JUDGE = Template("""You are a strict but fair semantic answer judge.

Question: $question
Gold answer(s): $correct_answers
Model answer: $model_answer

Rules:
- Judge only the final answer, not reasoning quality.
- Accept aliases, abbreviations, spelling variants, and equivalent entity names.
- Mark partial answers incorrect when the question asks for a more specific answer.
- For date/time answers, accept a coarser answer only if the question asks at that granularity.
- "I don't know", empty answers, or refusal to answer are Incorrect unless the gold answer explicitly says unknown.

Return exactly one token: Correct or Incorrect.""")

ALIAS_FRIENDLY_JUDGE = Template("""Decide whether the model answer is acceptable for the question.

Question: $question
Acceptable gold answers: $correct_answers
Model answer: $model_answer

Use an alias-friendly standard:
- Correct if the model answer refers to the same entity/value as any gold answer.
- Correct if it is a common short name, stage name, abbreviation, or harmless formatting variant.
- Incorrect if it names a different entity, relation, time, place, number, or gives only an unhelpful vague category.

Return exactly: Correct or Incorrect.""")

DECOMPOSED_FACT_JUDGE = Template("""Check whether the model answer satisfies the information need.

Question: $question
Gold answer(s): $correct_answers
Model answer: $model_answer

Procedure:
1. Identify the entity/relation/type of answer requested by the question.
2. Compare the model answer to the gold answer(s).
3. Ignore extra wording if the final answer is still unambiguous.

Do not explain your reasoning. Output exactly one word: Correct or Incorrect.""")


def _answers_to_text(answers: Any) -> str:
    if isinstance(answers, list):
        return "; ".join(str(a) for a in answers)
    return str(answers)


def build_judge_prompt(answers, model_answer: str):
    correct_answers = ", ".join(str(a) for a in answers)
    return LIST_ANSWER_JUDGE.substitute(
        correct_answers=correct_answers,
        model_answer=model_answer,
    )


def build_judge_prompts(answers, model_answer: str, question: str = "", variants: list[str] | None = None) -> list[dict[str, str]]:
    variants = variants or ["strict_semantic", "alias_friendly", "decomposed_fact"]
    correct_answers = _answers_to_text(answers)
    templates = {
        "legacy_list": LIST_ANSWER_JUDGE,
        "strict_semantic": STRICT_SEMANTIC_JUDGE,
        "alias_friendly": ALIAS_FRIENDLY_JUDGE,
        "decomposed_fact": DECOMPOSED_FACT_JUDGE,
    }
    prompts = []
    for name in variants:
        template = templates.get(name)
        if template is None:
            raise ValueError(f"Unknown judge prompt variant: {name}")
        kwargs = {
            "question": question or "",
            "correct_answers": correct_answers,
            "model_answer": model_answer or "",
        }
        prompts.append({"name": name, "prompt": template.substitute(**kwargs)})
    return prompts
