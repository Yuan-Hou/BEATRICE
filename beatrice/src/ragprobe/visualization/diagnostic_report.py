"""Compatibility wrapper for the rich probe diagnostic report.

New code should import ragprobe.visualization.rich_diagnostic_report directly.
This module remains so older scripts/configs that refer to diagnostic_report keep
getting the current report implementation.
"""

from ragprobe.visualization.rich_diagnostic_report import make_probe_report

__all__ = ["make_probe_report"]
