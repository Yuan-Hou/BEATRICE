"""Probe-checkpoint pinning: metadata recorded, mid-run change raises (spec §11)."""
import unittest

from pgir.scorer_client import ProbeCheckpointChanged, ScorerClient


class TestRunMetadata(unittest.TestCase):
    def _client(self):
        c = ScorerClient.__new__(ScorerClient)  # no network
        c.base_url = "http://x"
        c.concurrency = 16
        c.timeout_s = 180.0
        c.failures_path = None
        c.pinned = {"context": "ckpt_ctx.pth", "question": "ckpt_q.pth"}
        c._health = {"model_name": "Qwen/Qwen3.5-9B",
                     "probes": {"context": {"checkpoint": "ckpt_ctx.pth"},
                                "question": {"checkpoint": "ckpt_q.pth"}}}
        return c

    def test_metadata_contains_fingerprints(self):
        md = self._client().run_metadata()
        self.assertEqual(md["probe_checkpoints"]["context"], "ckpt_ctx.pth")
        self.assertEqual(md["probe_checkpoints"]["question"], "ckpt_q.pth")
        self.assertEqual(md["model_name"], "Qwen/Qwen3.5-9B")

    def test_matching_checkpoint_passes(self):
        self._client()._check_pin({"mode": "context", "probe_checkpoint": "ckpt_ctx.pth"})

    def test_fused_checkpoint_passes(self):
        # two-phase /score_context reports "ctx_ckpt + dual_ckpt" when fused
        c = self._client()
        c.dual_fusion = "dual_head.pth"
        c._check_pin({"mode": "context",
                      "probe_checkpoint": "ckpt_ctx.pth + dual_head.pth"})

    def test_fused_checkpoint_wrong_dual_raises(self):
        c = self._client()
        c.dual_fusion = "dual_head.pth"
        with self.assertRaises(ProbeCheckpointChanged):
            c._check_pin({"mode": "context",
                          "probe_checkpoint": "ckpt_ctx.pth + OTHER_dual.pth"})

    def test_changed_checkpoint_raises(self):
        with self.assertRaises(ProbeCheckpointChanged):
            self._client()._check_pin({"mode": "context", "probe_checkpoint": "OTHER.pth"})

    def test_unpinned_mode_ignored(self):
        c = self._client()
        c.pinned = {}
        c._check_pin({"mode": "context", "probe_checkpoint": "anything.pth"})


if __name__ == "__main__":
    unittest.main()
