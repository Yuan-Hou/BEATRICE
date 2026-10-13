"""Table-driven tests for the pure decision function (spec §11 test_decide)."""
import unittest

from pgir.arm import Arm
from pgir.pipeline import Abstain, Accept, Continue, PolicyCfg, ReturnArgmax, decide


def arm(p, status="success", answer="a"):
    return Arm(answer=answer, p_raw=p, p_cal=p, mode="context", status=status)


class TestDecide(unittest.TestCase):
    def test_table(self):
        cfg = PolicyCfg(tau_hi=0.8, budget=4, k=3)
        cfg_lo = PolicyCfg(tau_hi=0.8, budget=4, k=3, tau_lo=0.3)
        cases = [
            # (pool_ps, cfg, t, expected_type, expected_best_p)
            ([0.9], cfg, 0, Accept, 0.9),                   # early accept round 0
            ([0.8], cfg, 0, Accept, 0.8),                   # boundary: >= tau_hi accepts
            ([0.5], cfg, 0, Continue, None),                # low conf, budget left
            ([0.5, 0.79], cfg, 1, Continue, None),          # still below, keep going
            ([0.5, 0.85], cfg, 2, Accept, 0.85),            # accept mid-run on best arm
            ([0.5, 0.6, 0.7, 0.75], cfg, 3, ReturnArgmax, 0.75),   # budget exhausted
            ([0.7, 0.2, 0.3], cfg, 2, Continue, None),      # best is round-0, still continue
            ([0.1, 0.2], cfg, 3, ReturnArgmax, 0.2),        # no abstain when tau_lo unset
            ([0.1, 0.2], cfg_lo, 3, Abstain, None),         # abstain: best < tau_lo at budget
            ([0.5, 0.4], cfg_lo, 3, ReturnArgmax, 0.5),     # above tau_lo -> argmax
        ]
        for ps, c, t, expected, best_p in cases:
            with self.subTest(ps=ps, t=t, cfg=c):
                act = decide([arm(p) for p in ps], c, t)
                self.assertIsInstance(act, expected)
                if best_p is not None:
                    self.assertAlmostEqual(act.arm.p_cal, best_p)

    def test_argmax_is_pool_wide_not_last(self):
        cfg = PolicyCfg(tau_hi=0.99, budget=2)
        pool = [arm(0.7, answer="closed_book"), arm(0.3, answer="rag")]
        act = decide(pool, cfg, 1)
        self.assertIsInstance(act, ReturnArgmax)
        self.assertEqual(act.arm.answer, "closed_book")  # anti-regression fallback

    def test_failed_arms_ignored(self):
        cfg = PolicyCfg(tau_hi=0.8, budget=4)
        pool = [arm(0.99, status="failed"), arm(0.5)]
        self.assertIsInstance(decide(pool, cfg, 1), Continue)

    def test_all_failed(self):
        cfg = PolicyCfg(tau_hi=0.8, budget=4)
        self.assertIsInstance(decide([arm(0.9, status="failed")], cfg, 1), Continue)
        self.assertIsInstance(decide([arm(0.9, status="failed")], cfg, 3), Abstain)


if __name__ == "__main__":
    unittest.main()
