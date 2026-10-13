"""Static parity: gen_prompt reuses the service-side prompt code (spec §11).

Asserts (a) every prompt-touching symbol in pgir.gen_prompt IS the service's own
function (identity, not equality), (b) gen_prompt.py defines no prompt text of
its own, (c) the composed base prompt equals what the service-side pieces
produce for the same inputs.
"""
import inspect
import unittest

import ragprobe.features.extractor as extractor
import ragprobe.services.probe_scorer as probe_scorer
from ragprobe.prompts.qa import LEGACY_MUSIQUE_FEATURE_PROMPT, QUESTION_ONLY_FEATURE_PROMPT

import pgir.gen_prompt as gp


class TestGenPromptSource(unittest.TestCase):
    def test_symbols_are_service_side(self):
        self.assertIs(gp._build_prompt, extractor._build_prompt)
        self.assertIs(gp._apply_chat_template, extractor._apply_chat_template)
        self.assertIs(gp._format_context_legacy, extractor._format_context_legacy)
        self.assertIs(gp._normalize_contexts, probe_scorer._normalize_contexts)
        self.assertIs(gp.MODE_DEFAULTS, probe_scorer.MODE_DEFAULTS)

    def test_max_tokens_matches_scorer_default(self):
        self.assertEqual(gp.MAX_ANSWER_TOKENS, probe_scorer.MODE_DEFAULTS["context"]["answer_tokens"])
        self.assertEqual(gp.MAX_ANSWER_TOKENS, probe_scorer.MODE_DEFAULTS["question"]["answer_tokens"])

    def test_no_prompt_text_defined_here(self):
        src = inspect.getsource(gp)
        for marker in ("Directly answer", "based on your knowledge", "Template("):
            self.assertNotIn(marker, src, f"prompt text {marker!r} must not be defined in gen_prompt")

    def test_composition_matches_service_templates(self):
        q = "Who wrote Pride and Prejudice?"
        mode, prompt = gp.build_base_prompt(q, None)
        self.assertEqual(mode, "question")
        self.assertEqual(prompt, QUESTION_ONLY_FEATURE_PROMPT.substitute(question=q))

        ctxs = [{"text": "Pride and Prejudice is an 1813 novel by Jane Austen."},
                "It was published anonymously."]
        mode, prompt = gp.build_base_prompt(q, ctxs)
        self.assertEqual(mode, "context")
        expected_ctx = ("- Pride and Prejudice is an 1813 novel by Jane Austen.\n"
                        "- It was published anonymously.")
        self.assertEqual(prompt, LEGACY_MUSIQUE_FEATURE_PROMPT.substitute(context=expected_ctx,
                                                                          question=q))

    def test_mode_rule_matches_scorer(self):
        self.assertEqual(gp.resolve_mode(None), "question")
        self.assertEqual(gp.resolve_mode([]), "question")
        self.assertEqual(gp.resolve_mode([{"text": "x"}]), "context")


if __name__ == "__main__":
    unittest.main()
