"""Hallucination / answer-correctness detection baselines.

These reproduce simple white-box baselines (logprob/entropy uncertainty, SAPLMA
hidden-state probe, LLM-Check training-free statistics) on top of the *already
extracted* per-token features, so they share the exact same train/val scope and
binary labels as the main probes. See docs/baselines.md for the method list.
"""

from ragprobe.baselines.extract import extract_split_signals, ExampleSignals
from ragprobe.baselines.evaluate import binary_detection_metrics, evaluate_prediction_files

__all__ = [
    "extract_split_signals",
    "ExampleSignals",
    "binary_detection_metrics",
    "evaluate_prediction_files",
]
