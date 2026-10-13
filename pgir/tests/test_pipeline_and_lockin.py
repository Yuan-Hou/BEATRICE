"""Control-loop tests with fake evaluator/retriever: anti-regression argmax,
lock-in S2->S1 fallback (spec §11 test_lockin_fallback), cost accounting."""
import unittest

from pgir.arm import Arm, Cost
from pgir.pipeline import PGIR, PolicyCfg
from pgir.strategies import EvidenceState, next_evidence


class FakeEvaluator:
    """Deterministic p schedule: closed-book then per-round context scores."""

    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.calls = 0

    def evaluate(self, question, contexts=None, example_id=None,
                 evidence_pids=None, round_index=-1):
        p = self.schedule[min(self.calls, len(self.schedule) - 1)]
        self.calls += 1
        return Arm(answer=f"ans{self.calls - 1}", p_raw=p, p_cal=p,
                   mode="context" if contexts else "question",
                   cost=Cost(gen_calls=1, score_calls=1),
                   round_index=round_index, evidence_pids=evidence_pids or [],
                   contexts=contexts or [])


class FakeRetriever:
    """Returns docs d{offset}..; `repeat=True` always returns the same page."""

    def __init__(self, repeat=False):
        self.repeat = repeat
        self.queries = []

    def search(self, query, num):
        self.queries.append((query, num))
        if self.repeat:
            return [{"id": f"d{i}", "contents": f"passage {i}"} for i in range(num)]
        base = len(self.queries) * 100
        return [{"id": f"d{base + i}", "contents": f"passage {base + i}"} for i in range(num)]


class TestPipeline(unittest.TestCase):
    def test_early_accept_round0_skips_retrieval(self):
        ev, rt = FakeEvaluator([0.95]), FakeRetriever()
        res = PGIR(ev, rt, PolicyCfg(tau_hi=0.8, budget=4)).run_item("q", "e1")
        self.assertEqual(res.action, "accept")
        self.assertEqual(res.stop_round, 0)
        self.assertEqual(res.cost.retr_calls, 0)
        self.assertEqual(res.cost.gen_calls, 1)

    def test_accept_mid_run(self):
        ev, rt = FakeEvaluator([0.4, 0.5, 0.9]), FakeRetriever()
        res = PGIR(ev, rt, PolicyCfg(tau_hi=0.8, budget=4)).run_item("q")
        self.assertEqual(res.action, "accept")
        self.assertEqual(res.stop_round, 2)
        self.assertEqual(len(res.pool), 3)

    def test_anti_regression_argmax_returns_closed_book(self):
        # retrieval keeps making things worse; final answer must be the round-0 arm
        ev, rt = FakeEvaluator([0.6, 0.3, 0.2, 0.1]), FakeRetriever()
        res = PGIR(ev, rt, PolicyCfg(tau_hi=0.9, budget=4)).run_item("q")
        self.assertEqual(res.action, "argmax")
        self.assertEqual(res.answer, "ans0")
        self.assertEqual(res.pool[0].mode, "question")

    def test_full_budget_never_stops_early(self):
        ev, rt = FakeEvaluator([0.99, 0.99, 0.99, 0.99]), FakeRetriever()
        res = PGIR(ev, rt, PolicyCfg(tau_hi=0.8, budget=4), full_budget=True).run_item("q")
        self.assertEqual(len(res.pool), 4)  # 1 closed-book + 3 evidence rounds
        self.assertEqual(res.action, "accept")  # earliest accept recorded for diagnostics

    def test_abstain(self):
        ev, rt = FakeEvaluator([0.1, 0.1, 0.1, 0.1]), FakeRetriever()
        res = PGIR(ev, rt, PolicyCfg(tau_hi=0.9, budget=2, tau_lo=0.3)).run_item("q")
        self.assertEqual(res.action, "abstain")
        self.assertEqual(res.answer, "")

    def test_cost_accounting(self):
        ev, rt = FakeEvaluator([0.1, 0.2, 0.3, 0.4]), FakeRetriever()
        res = PGIR(ev, rt, PolicyCfg(tau_hi=0.9, budget=3)).run_item("q")
        # rounds: closed-book + 2 evidence arms = 3 gen + 3 score; retr: initial + 1 next_evidence
        self.assertEqual(res.cost.gen_calls, 3)
        self.assertEqual(res.cost.score_calls, 3)
        self.assertEqual(res.cost.retr_calls, 2)


class TestLockinFallback(unittest.TestCase):
    def _arm(self, p, answer="a"):
        return Arm(answer=answer, p_raw=p, p_cal=p, mode="context")

    def test_high_overlap_forces_s1(self):
        rt = FakeRetriever(repeat=True)  # every search returns identical pids
        state = EvidenceState()
        cfg = PolicyCfg(k=3)
        state.register(rt.search("q", 3))          # round 1 evidence
        pool = [self._arm(0.4)]
        docs, _ = next_evidence("q", pool, state.round_pids[-1], rt, state, cfg, "s2")
        # S2 dedup already avoids used pids; overlap recorded for the new round
        self.assertEqual(len(state.round_pids), 2)
        # force full overlap to trip the guard: register the same page again
        state.overlaps[-1] = 1.0
        before = state.forced_s1
        docs, _ = next_evidence("q", pool, docs, rt, state, cfg, "s2")
        self.assertEqual(state.forced_s1, before + 1)  # S2 -> S1 downgrade fired
        self.assertTrue(all(q == "q" for q, _ in rt.queries[-1:]))  # S1 uses original query

    def test_s2_query_contains_answer(self):
        rt = FakeRetriever()
        state = EvidenceState()
        cfg = PolicyCfg(k=3)
        state.register(rt.search("q", 3))
        pool = [self._arm(0.4, answer="Jane Austen")]
        next_evidence("who wrote it?", pool, [], rt, state, cfg, "s2")
        q, _ = rt.queries[-1]
        self.assertIn("Jane Austen", q)

    def test_diagnostics_fields(self):
        state = EvidenceState()
        state.register([{"id": "a"}, {"id": "b"}])
        state.register([{"id": "a"}, {"id": "c"}])
        pool = [self._arm(0.4), self._arm(0.405), self._arm(0.5)]
        d = state.diagnostics(pool)
        self.assertAlmostEqual(d["overlaps"][0], 1 / 3)
        self.assertIn("score_plateau_rate", d)
        self.assertAlmostEqual(d["score_plateau_rate"], 0.5)  # |Δ|=.005 (<.02), |Δ|=.095


if __name__ == "__main__":
    unittest.main()
