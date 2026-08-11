"""
Statistical anomaly detection via single-sample NIS hypothesis testing.

Reuses M5's chi-squared machinery (compute_nis, the same theoretical
distribution), but asks a different question: not "is the filter
well-calibrated on average" (M5, two-sided, many-sample test), but "is
THIS ONE measurement more surprising than nominal noise would explain"
(one-sided, single-sample test).

Threshold: chi2_(1-alpha)(m) -- the upper (1-alpha) quantile of the
chi-squared distribution with m degrees of freedom (m = measurement
dimension). One-sided because a dynamical deviation (unplanned Delta-v,
unmodeled perturbation) INCREASES innovation/NIS; nothing in this
project's scenarios makes a measurement suspiciously too accurate, so
only an upper bound is meaningful here.

Honest limitation: even a perfectly consistent filter will exceed this
threshold at rate alpha by pure chance -- this is not a flaw, it is the
definition of a single-sample hypothesis test at significance alpha.
detect_with_persistence() exists specifically to reduce this false-alarm
rate, at the cost of some detection latency, by requiring several
flagged samples within a recent window before confirming an anomaly.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def nis_threshold(dof: int, alpha: float = 0.01) -> float:
    """
    One-sided upper threshold for single-sample NIS anomaly flagging.

    Args:
        dof: degrees of freedom (measurement dimension, m=3 here).
        alpha: false-alarm rate under nominal conditions (0.01 -> a
            perfectly consistent filter still flags ~1% of nominal
            measurements by chance).

    Returns:
        The NIS value above which a single measurement is flagged.
    """
    return float(chi2.ppf(1 - alpha, df=dof))


def is_single_sample_anomalous(nis_value: float, threshold: float) -> bool:
    """Raw single-sample flag: True if this one NIS value exceeds the threshold."""
    return bool(nis_value > threshold)


def detect_with_persistence(
    nis_series: np.ndarray, threshold: float, required_hits: int = 3, window: int = 5
) -> np.ndarray:
    """
    M-of-N persistence-based confirmed-anomaly detection.

    A CONFIRMED anomaly is flagged at index k only if at least
    required_hits of the most recent `window` single-sample flags
    (including k itself) are True. This filters out isolated single-
    sample false alarms (expected at rate alpha even when nominal)
    while still detecting a genuine, sustained deviation within a few
    samples of it starting.

    Args:
        nis_series: (N,) array of NIS values over time.
        threshold: single-sample flag threshold (from nis_threshold).
        required_hits: minimum flagged samples within the window to confirm.
        window: number of most-recent samples considered.

    Returns:
        (N,) boolean array, True where a CONFIRMED (persistent) anomaly
        is declared at that index.
    """
    single_sample_flags = nis_series > threshold
    n = len(nis_series)
    confirmed = np.zeros(n, dtype=bool)

    for k in range(n):
        window_start = max(0, k - window + 1)
        hits_in_window = np.sum(single_sample_flags[window_start : k + 1])
        confirmed[k] = hits_in_window >= required_hits

    return confirmed
