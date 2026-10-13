"""Calibration fits only on validation; test/OOD fits raise (spec §11)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pgir.calibration import (CalibratorProbeMismatch, PerModeCalibrator,
                              TestSplitViolation, brier, ece)


def _synth(n, seed=17):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(size=n) < p ** 1.5).astype(int)  # miscalibrated on purpose
    return p, y


class TestCalibrationDiscipline(unittest.TestCase):
    def test_fit_on_test_raises(self):
        p, y = _synth(100)
        for split in ("test", "TEST", "hotpotqa_test", "ood", "popqa"):
            with self.subTest(split=split):
                with self.assertRaises(TestSplitViolation):
                    PerModeCalibrator().fit(p, y, "context", split)

    def test_fit_on_validation_ok(self):
        p, y = _synth(500)
        c = PerModeCalibrator().fit(p, y, "context", "validation")
        self.assertEqual(c.meta["context"]["kind"], "platt")  # n<2000 -> Platt
        p2, y2 = _synth(5000)
        c.fit(p2, y2, "question", "dev")
        self.assertEqual(c.meta["question"]["kind"], "isotonic")

    def test_calibration_improves_ece(self):
        p, y = _synth(5000)
        c = PerModeCalibrator().fit(p, y, "context", "validation")
        pc = c.calibrate(p, "context")
        self.assertLess(ece(pc, y), ece(p, y))
        self.assertLessEqual(brier(pc, y), brier(p, y) + 1e-9)

    def test_identity_before_fit_and_scalar_api(self):
        c = PerModeCalibrator()
        self.assertEqual(c.calibrate(0.42, "context"), 0.42)
        p, y = _synth(3000)
        c.fit(p, y, "context", "validation")
        v = c.calibrate(0.42, "context")
        self.assertIsInstance(v, float)
        self.assertTrue(0.0 <= v <= 1.0)

    def test_monotone(self):
        p, y = _synth(5000)
        c = PerModeCalibrator().fit(p, y, "context", "validation")
        grid = np.linspace(0, 1, 101)
        out = c.calibrate(grid, "context")
        self.assertTrue(np.all(np.diff(out) >= -1e-9))

    def test_provenance_guard(self):
        p, y = _synth(3000)
        c = PerModeCalibrator().fit(p, y, "context", "validation",
                                    probe_checkpoint="ckptA.pth")
        # matching probe: no raise
        c.check_provenance({"context": "ckptA.pth"}, strict=True)
        # served a different probe: must raise
        with self.assertRaises(CalibratorProbeMismatch):
            c.check_provenance({"context": "ckptB.pth"}, strict=True)
        # provenance survives save/load
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cal.json"
            c.save(path)
            c2 = PerModeCalibrator.load(path)
            self.assertEqual(c2.meta["context"]["probe_checkpoint"], "ckptA.pth")
            with self.assertRaises(CalibratorProbeMismatch):
                c2.check_provenance({"context": "ckptB.pth"}, strict=True)

    def test_provenance_unknown_does_not_raise(self):
        # older calibrator with no recorded provenance: cannot prove mismatch
        p, y = _synth(3000)
        c = PerModeCalibrator().fit(p, y, "context", "validation")
        c.check_provenance({"context": "anything.pth"}, strict=True)  # no raise

    def test_save_load_roundtrip(self):
        p, y = _synth(3000)
        c = PerModeCalibrator().fit(p, y, "context", "validation")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cal.json"
            c.save(path)
            c2 = PerModeCalibrator.load(path)
            grid = np.linspace(0, 1, 51)
            np.testing.assert_allclose(c.calibrate(grid, "context"),
                                       c2.calibrate(grid, "context"), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
