"""
Filter consistency testing: NEES and NIS, via the chi-squared distribution.

See docs/mathematics.md for the full derivation of why e^T @ P^-1 @ e
(the Mahalanobis distance squared) follows a chi-squared distribution
when e is genuinely Gaussian with covariance P.

    NEES (Normalized Estimation Error Squared):
        NEES_k = e_k^T @ P_k^-1 @ e_k,  e_k = x_true_k - x_hat_k
        Requires ground truth -- simulation only. Expected value = n
        (state dimension) if the filter is consistent.

    NIS (Normalized Innovation Squared):
        NIS_k = nu_k^T @ S_k^-1 @ nu_k
        Uses only the filter's own innovation/S -- no ground truth
        needed, so this generalizes to real deployed systems. Expected
        value = m (measurement dimension) if consistent.

Methodological limitation, stated explicitly: the confidence intervals
here treat consecutive TIME SAMPLES from a single trajectory as if they
were independent draws, which is an approximation (EKF errors are
temporally correlated within one run). The textbook-rigorous approach
uses independent MONTE CARLO trials instead -- that is stretch goal S2.
This module's time-averaged check is a common, practically useful, but
not fully rigorous first-pass consistency test.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def compute_nees(true_state: np.ndarray, estimated_state: np.ndarray, P: np.ndarray) -> float:
    """
    Normalized Estimation Error Squared for one time step.

    Requires ground truth -- only computable in simulation, never in a
    real deployed filter.
    """
    e = true_state - estimated_state
    return float(e.T @ np.linalg.inv(P) @ e)


def compute_nis(innovation: np.ndarray, S: np.ndarray) -> float:
    """
    Normalized Innovation Squared for one time step.

    Uses only the filter's own innovation and innovation covariance --
    no ground truth needed. This is the statistic that generalizes to
    M6's anomaly detection.
    """
    return float(innovation.T @ np.linalg.inv(S) @ innovation)


def consistency_interval(dof: int, n_samples: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Two-sided (1-alpha) confidence interval for the TIME-AVERAGED
    NEES or NIS over n_samples samples, under the (approximate, see
    module docstring) assumption that samples are independent.

    Args:
        dof: degrees of freedom of the underlying chi-squared
            distribution (n=6 for NEES, m=3 for NIS in this project).
        n_samples: number of samples being averaged.
        alpha: significance level (0.05 -> 95% confidence interval).

    Returns:
        (lower, upper) bounds for the sample average. A time-averaged
        NEES/NIS value inside this interval does not reject filter
        consistency at this significance level; outside indicates
        likely inconsistency (optimistic if average is too HIGH,
        conservative if too LOW).
    """
    lower = chi2.ppf(alpha / 2, df=n_samples * dof) / n_samples
    upper = chi2.ppf(1 - alpha / 2, df=n_samples * dof) / n_samples
    return lower, upper
