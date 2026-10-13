"""Per-mode probability calibration (spec §8.2).

- isotonic regression per mode ("question" / "context"), datasets pooled;
- falls back to Platt scaling when n < 2000 (isotonic overfits small val);
- refuses to fit on anything labelled test/ood — validation only, enforced
  (tests/test_calibration_discipline.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FORBIDDEN_SPLITS = ("test", "ood", "popqa")
MIN_ISOTONIC_N = 2000


class TestSplitViolation(RuntimeError):
    pass


class CalibratorProbeMismatch(RuntimeError):
    """Raised when a loaded calibrator was fit against a different probe
    checkpoint than the one the scorer service is currently serving.

    A per-mode calibrator maps THIS probe's raw scores -> P(correct). Swap the
    probe (e.g. a retrained checkpoint served over the same HTTP endpoint) and
    the old isotonic/Platt map silently produces wrong p_cal and a mis-set
    accept threshold. This guard forces a recalibration instead."""
    pass


def _check_split(split: str) -> None:
    s = str(split).lower()
    if any(bad in s for bad in FORBIDDEN_SPLITS):
        raise TestSplitViolation(
            f"calibration fit on split '{split}' is forbidden — validation only")


class _Platt:
    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(self._logit(np.asarray(p, float)).reshape(-1, 1), np.asarray(y, int))
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        return self

    def predict(self, p):
        z = self.a * self._logit(np.asarray(p, float)) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self):
        return {"kind": "platt", "a": self.a, "b": self.b}


class _Isotonic:
    def __init__(self, x=None, y=None):
        self.x, self.y = x, y

    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(np.asarray(p, float), np.asarray(y, int))
        # store the step function itself (portable, no sklearn needed to load)
        self.x = [float(v) for v in iso.X_thresholds_]
        self.y = [float(v) for v in iso.y_thresholds_]
        return self

    def predict(self, p):
        return np.interp(np.asarray(p, float), self.x, self.y)

    def to_dict(self):
        return {"kind": "isotonic", "x": self.x, "y": self.y}


def _from_dict(d):
    if d["kind"] == "platt":
        return _Platt(d["a"], d["b"])
    return _Isotonic(d["x"], d["y"])


class PerModeCalibrator:
    def __init__(self):
        self.models: dict[str, object] = {}
        self.meta: dict[str, dict] = {}

    def fit(self, p_raw, labels, mode: str, split: str,
            probe_checkpoint: str | None = None) -> "PerModeCalibrator":
        _check_split(split)
        p = np.asarray(p_raw, float)
        y = np.asarray(labels, int)
        if p.shape[0] != y.shape[0] or p.shape[0] == 0:
            raise ValueError("p_raw and labels must be same nonzero length")
        kind = "isotonic" if p.shape[0] >= MIN_ISOTONIC_N else "platt"
        model = (_Isotonic() if kind == "isotonic" else _Platt()).fit(p, y)
        self.models[mode] = model
        # probe_checkpoint records WHICH probe this map was fit against, so a
        # later run can refuse to silently apply it to a different probe.
        self.meta[mode] = {"kind": kind, "n": int(p.shape[0]), "split": split,
                           "probe_checkpoint": probe_checkpoint}
        return self

    def check_provenance(self, pinned: dict[str, str], strict: bool = True) -> None:
        """Verify each fitted mode was calibrated against the probe checkpoint
        the scorer now serves (`pinned` = ScorerClient.pinned, mode->checkpoint).

        strict=True raises CalibratorProbeMismatch on any disagreement; a mode
        with no recorded provenance (older calibrator) is reported but does not
        raise, since we cannot prove a mismatch."""
        problems, unknown = [], []
        for mode in self.models:
            fit_ckpt = (self.meta.get(mode) or {}).get("probe_checkpoint")
            served = pinned.get(mode)
            if fit_ckpt is None:
                unknown.append(mode)
                continue
            if served is not None and fit_ckpt != served:
                problems.append(f"mode={mode}: calibrator fit on {fit_ckpt} "
                                f"but scorer serves {served}")
        if problems and strict:
            raise CalibratorProbeMismatch(
                "calibrator does not match the served probe — recalibrate on the "
                "new probe (pgir.tuning.run_tuning). Details: " + "; ".join(problems))
        return None

    def calibrate(self, p, mode: str):
        if mode not in self.models:
            return p  # identity until fitted (raw probe output)
        out = self.models[mode].predict(np.atleast_1d(np.asarray(p, float)))
        return float(out[0]) if np.isscalar(p) or getattr(p, "ndim", 0) == 0 else out

    # ------------------------------------------------------------ persistence
    def save(self, path):
        payload = {m: {"model": self.models[m].to_dict(), "meta": self.meta[m]}
                   for m in self.models}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path) -> "PerModeCalibrator":
        payload = json.loads(Path(path).read_text())
        c = cls()
        for mode, entry in payload.items():
            c.models[mode] = _from_dict(entry["model"])
            c.meta[mode] = entry.get("meta", {})
        return c


# ------------------------------------------------------------ diagnostics ----
def ece(p, y, n_bins: int = 15) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier(p, y) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def reliability_table(p, y, n_bins: int = 15) -> list[dict]:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
                     "p_mean": float(p[m].mean()) if m.sum() else math.nan,
                     "y_rate": float(y[m].mean()) if m.sum() else math.nan})
    return rows
