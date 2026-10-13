"""Best-of-N (SELECT) + STEER tests: adaptive band, dedup, cost accounting,
decide_steer table, rewrite fallback, round-aware trace replay.
Defaults-off backward compat is asserted by test_pipeline_and_lockin passing
unchanged with fakes that lack vllm/evaluate_many entirely."""
import unittest

from pgir.arm import Arm, Cost
from pgir.pipeline import PGIR, PolicyCfg
from pgir.steer import build_rewrite_prompt, clean_rewrite, decide_steer


class FakeVLLM:
    def __init__(self, samples=None, rewrite="alt query"):
        self.samples = list(samples or [])
        self.rewrite = rewrite
        self.sample_calls = []
        self.free_calls = []

    def generate_samples(self, question, contexts=None, n=3, temperature=0.8):
        self.sample_calls.append((question, n))
        return list(self.samples[:n])

    def generate_free(self, prompt, max_tokens=64, temperature=0.0):
        self.free_calls.append(prompt)
        return self.rewrite


class FakeEvaluator:
    """evaluate: p schedule for greedy arms; evaluate_many: p by answer map."""

    def __init__(self, schedule, sampled_p=None, vllm=None):
        self.schedule = list(schedule)
        self.sampled_p = dict(sampled_p or {})
        self.vllm = vllm
        self.calls = 0

    def _arm(self, p, answer, mode, t, source=""):
        return Arm(answer=answer, p_raw=p, p_cal=p, mode=mode,
                   answer_source=source, cost=Cost(gen_calls=1, score_calls=1),
                   round_index=t)

    def evaluate(self, question, contexts=None, example_id=None,
                 evidence_pids=None, round_index=-1):
        p = self.schedule[min(self.calls, len(self.schedule) - 1)]
        self.calls += 1
        return self._arm(p, f"ans{self.calls - 1}",
                         "context" if contexts else "question", round_index)

    def evaluate_many(self, jobs):
        return [self._arm(self.sampled_p.get(j["answer"], 0.1), j["answer"],
                          "context" if j.get("contexts") else "question",
                          j.get("round_index", -1), j.get("answer_source", ""))
                for j in jobs]


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, num):
        self.queries.append(query)
        base = len(self.queries) * 100
        return [{"id": f"d{base + i}", "contents": f"p{base + i}"} for i in range(num)]


BON = dict(bon_n=3, bon_lo=0.05, tau_hi=0.8, budget=4)


class TestBestOfN(unittest.TestCase):
    def test_no_sampling_outside_band(self):
        for greedy_p in (0.95, 0.02):  # above tau_hi / below bon_lo
            vllm = FakeVLLM(samples=["s1", "s2"])
            ev = FakeEvaluator([greedy_p], vllm=vllm)
            PGIR(ev, FakeRetriever(), PolicyCfg(**BON)).run_item("q")
            self.assertEqual(vllm.sample_calls, [], f"greedy_p={greedy_p}")

    def test_in_band_samples_join_pool_and_can_accept(self):
        # greedy 0.4 (in band) -> sample; one sampled candidate scores 0.9 -> accept round 0
        vllm = FakeVLLM(samples=["good", "bad"])
        ev = FakeEvaluator([0.4], sampled_p={"good": 0.9, "bad": 0.2}, vllm=vllm)
        res = PGIR(ev, FakeRetriever(), PolicyCfg(**BON)).run_item("q")
        self.assertEqual(res.action, "accept")
        self.assertEqual(res.stop_round, 0)
        self.assertEqual(res.answer, "good")
        sources = [a.answer_source for a in res.pool]
        self.assertEqual(sources.count("sampled"), 2)
        # cost: 3 arms = 3 gen + 3 score, no retrieval
        self.assertEqual(res.cost.gen_calls, 3)
        self.assertEqual(res.cost.score_calls, 3)
        self.assertEqual(res.cost.retr_calls, 0)

    def test_dedup_drops_greedy_duplicates_and_empties(self):
        # both samples collapse onto the round-0 greedy answer (case/ws) or empty;
        # budget=1 keeps the run to round 0 where greedy == "ans0"
        vllm = FakeVLLM(samples=["  ANS0 ", ""])
        ev = FakeEvaluator([0.4], vllm=vllm)
        res = PGIR(ev, FakeRetriever(),
                   PolicyCfg(bon_n=3, bon_lo=0.05, tau_hi=0.8, budget=1)).run_item("q")
        self.assertEqual(vllm.sample_calls, [("q", 2)])  # sampling did fire
        self.assertEqual([a for a in res.pool if a.answer_source == "sampled"], [])

    def test_bon_off_never_touches_vllm_sampling(self):
        vllm = FakeVLLM(samples=["s1"])
        ev = FakeEvaluator([0.4, 0.4, 0.4, 0.4], vllm=vllm)
        res = PGIR(ev, FakeRetriever(), PolicyCfg(tau_hi=0.8, budget=4)).run_item("q")
        self.assertEqual(vllm.sample_calls, [])
        self.assertEqual(len(res.pool), 4)  # unchanged legacy shape


def _arm(p, mode="context", t=1, status="success", answer="a"):
    return Arm(answer=answer, p_raw=p, p_cal=p, mode=mode, status=status,
               round_index=t)


class TestDecideSteer(unittest.TestCase):
    CFG = PolicyCfg(tau_hi=0.8, steer=True, steer_delta=0.15, steer_eps=0.02)

    def test_no_context_rounds_default_s2(self):
        self.assertEqual(decide_steer([_arm(0.6, "question", 0)], self.CFG), "s2")

    def test_lockin_guard_s1(self):
        pool = [_arm(0.6, "question", 0), _arm(0.3, "context", 1)]
        self.assertEqual(decide_steer(pool, self.CFG), "s1")

    def test_stagnation_rewrite(self):
        pool = [_arm(0.3, "question", 0),
                _arm(0.35, "context", 1), _arm(0.355, "context", 2)]
        self.assertEqual(decide_steer(pool, self.CFG), "rewrite")

    def test_improving_defaults_s2(self):
        pool = [_arm(0.3, "question", 0),
                _arm(0.35, "context", 1), _arm(0.5, "context", 2)]
        self.assertEqual(decide_steer(pool, self.CFG), "s2")

    def test_round_best_uses_max_over_siblings(self):
        # round 1 has a weak greedy but a strong sampled sibling -> no lock-in guard
        pool = [_arm(0.6, "question", 0),
                _arm(0.3, "context", 1), _arm(0.7, "context", 1)]
        self.assertEqual(decide_steer(pool, self.CFG), "s2")

    def test_failed_arms_ignored(self):
        pool = [_arm(0.6, "question", 0), _arm(0.9, "context", 1, status="failed"),
                _arm(0.3, "context", 1)]
        self.assertEqual(decide_steer(pool, self.CFG), "s1")


class TestSteerIntegration(unittest.TestCase):
    def test_lockin_routes_to_s1_original_query(self):
        # q0=.6, ctx rounds all .3 -> steer picks s1 -> retriever gets original q
        ev = FakeEvaluator([0.6, 0.3, 0.3, 0.3], vllm=FakeVLLM())
        rt = FakeRetriever()
        res = PGIR(ev, rt, PolicyCfg(tau_hi=0.9, budget=4, steer=True)).run_item("myq")
        self.assertIn("s1", res.diagnostics["steer_log"])
        self.assertTrue(any(q == "myq" for q in rt.queries[1:]))

    def test_stagnation_rewrite_uses_rewritten_query(self):
        vllm = FakeVLLM(rewrite="totally different query")
        ev = FakeEvaluator([0.3, 0.35, 0.355, 0.36], vllm=vllm)
        rt = FakeRetriever()
        res = PGIR(ev, rt, PolicyCfg(tau_hi=0.9, budget=4, steer=True)).run_item("myq")
        self.assertIn("rewrite", res.diagnostics["steer_log"])
        self.assertIn("totally different query", rt.queries)
        self.assertEqual(len(vllm.free_calls), 1)
        # the rewrite gen call is charged
        self.assertEqual(res.cost.gen_calls, 4 + 1)

    def test_degenerate_rewrite_falls_back_to_s1(self):
        vllm = FakeVLLM(rewrite="myq")  # echoes the question -> degenerate
        ev = FakeEvaluator([0.3, 0.35, 0.355, 0.36], vllm=vllm)
        rt = FakeRetriever()
        res = PGIR(ev, rt, PolicyCfg(tau_hi=0.9, budget=4, steer=True)).run_item("myq")
        self.assertIn("s1", res.diagnostics["steer_log"])

    def test_steer_off_logs_nothing(self):
        ev = FakeEvaluator([0.6, 0.3, 0.3, 0.3], vllm=FakeVLLM())
        res = PGIR(ev, FakeRetriever(), PolicyCfg(tau_hi=0.9, budget=4)).run_item("q")
        self.assertEqual(res.diagnostics["steer_log"], [])


class TestSteerHelpers(unittest.TestCase):
    def test_rewrite_prompt_dedups_and_caps_answers(self):
        p = build_rewrite_prompt("q?", ["A b", "a  B", "c", "d", "e", "f"])
        self.assertEqual(p.count("A b"), 1)
        self.assertNotIn("; f", p)

    def test_clean_rewrite(self):
        self.assertEqual(clean_rewrite('  "new query"\njunk', "q?"), "new query")
        self.assertEqual(clean_rewrite("Q?", "q?"), "")   # echo -> degenerate
        self.assertEqual(clean_rewrite("\n\n", "q?"), "")


class TestRoundAwareSimulate(unittest.TestCase):
    def test_sampled_sibling_can_accept_at_round0(self):
        from pgir.tuning.trace import simulate

        item = {"example_id": "e", "pool": [
            {"round_index": 0, "status": "success", "mode": "question",
             "p_raw": 0.4, "p_cal": 0.4, "answer": "greedy", "timing": {}},
            {"round_index": 0, "status": "success", "mode": "question",
             "p_raw": 0.9, "p_cal": 0.9, "answer": "sampled", "timing": {}},
            {"round_index": 1, "status": "success", "mode": "context",
             "p_raw": 0.5, "p_cal": 0.5, "answer": "later", "timing": {}},
        ], "diagnostics": {"retr_log": [1], "retr_times_s": [0.1]}}
        sim = simulate(item, PolicyCfg(tau_hi=0.8, budget=4))
        self.assertEqual(sim["action"], "accept")
        self.assertEqual(sim["stop_round"], 0)
        self.assertEqual(sim["answer"], "sampled")
        self.assertEqual(sim["gen_calls"], 2)   # both round-0 arms consumed
        self.assertEqual(sim["retr_calls"], 0)  # stopped before evidence


if __name__ == "__main__":
    unittest.main()
