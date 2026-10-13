"""Full-budget trace replay == live run, field by field (spec §11).

Uses deterministic fake evaluator/retriever pairs: one PGIR live run with early
stopping, one full-budget trace run, then `simulate` on the trace must
reproduce the live decisions, answers and costs for the same policy — across a
grid of policies and score schedules.
"""
import unittest

from pgir.pipeline import PGIR, PolicyCfg
from pgir.tests.test_pipeline_and_lockin import FakeEvaluator, FakeRetriever
from pgir.tuning.trace import simulate


def run_pair(schedule, cfg):
    live = PGIR(FakeEvaluator(schedule), FakeRetriever(), cfg).run_item("q", "e1")
    full = PGIR(FakeEvaluator(schedule), FakeRetriever(),
                PolicyCfg(tau_hi=cfg.tau_hi, budget=4, k=cfg.k, tau_lo=cfg.tau_lo),
                full_budget=True).run_item("q", "e1")
    sim = simulate(full.to_dict(), cfg)
    return live, sim


class TestSimulateConsistency(unittest.TestCase):
    def test_grid(self):
        schedules = [
            [0.95, 0.1, 0.1, 0.1],   # accept at round 0
            [0.4, 0.9, 0.1, 0.1],    # accept at round 1
            [0.4, 0.5, 0.6, 0.85],   # accept at last round
            [0.4, 0.5, 0.3, 0.2],    # argmax fallback (best = round 2)
            [0.6, 0.3, 0.2, 0.1],    # anti-regression: round-0 argmax
            [0.05, 0.1, 0.1, 0.1],   # abstain-eligible lows
        ]
        cfgs = [PolicyCfg(tau_hi=0.8, budget=b, k=3) for b in (2, 3, 4)]
        cfgs += [PolicyCfg(tau_hi=0.7, budget=3, k=3, tau_lo=0.3)]
        for schedule in schedules:
            for cfg in cfgs:
                with self.subTest(schedule=schedule, cfg=cfg):
                    live, sim = run_pair(schedule, cfg)
                    self.assertEqual(sim["answer"], live.answer)
                    self.assertEqual(sim["action"], live.action)
                    self.assertEqual(sim["stop_round"], live.stop_round)
                    self.assertEqual(sim["gen_calls"], live.cost.gen_calls)
                    self.assertEqual(sim["score_calls"], live.cost.score_calls)
                    self.assertEqual(sim["retr_calls"], live.cost.retr_calls)


if __name__ == "__main__":
    unittest.main()
