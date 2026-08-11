"""
Tests for NEES/NIS filter consistency checking.

Split across:
    - Isolated correctness of the NEES/NIS math itself, using synthetic
      Gaussian samples with a KNOWN true covariance (no EKF involved) --
      same isolation-testing spirit as M1/M3 testing their own math
      independent of the rest of the pipeline.
    - The real EKF's actual consistency over the nominal scenario.
    - A deliberately BROKEN filter (wrong R) to prove the consistency
      test actually has teeth -- it must correctly FAIL, not just
      always pass regardless of input.
"""

import numpy as np
import pytest

from orbitdet.dynamics.integrators import propagate_rk4
from orbitdet.dynamics.two_body import two_body_eom
from orbitdet.estimation.consistency import compute_nees, compute_nis, consistency_interval
from orbitdet.estimation.ekf import ekf_predict, ekf_update
from orbitdet.measurements.models import generate_measurement_series, position_measurement_matrix

MU_EARTH = 398600.4418
R0 = np.array([6778.137, 0.0, 0.0])
V0 = np.array([0.0, 7.6685582, 0.0])
STATE0 = np.concatenate([R0, V0])


def test_nees_is_zero_for_perfect_estimate():
    """If the estimate exactly equals truth, the error is zero, so NEES must be exactly zero."""
    P = np.eye(6) * 0.5
    nees = compute_nees(STATE0, STATE0.copy(), P)
    assert nees == pytest.approx(0.0, abs=1e-10)


def test_nis_is_zero_for_zero_innovation():
    """If the innovation is exactly zero (measurement matched prediction perfectly), NIS must be zero."""
    S = np.eye(3) * 0.25
    nis = compute_nis(np.zeros(3), S)
    assert nis == pytest.approx(0.0, abs=1e-10)


def test_nees_matches_synthetic_gaussian_samples_with_known_covariance():
    """
    Isolated correctness check: generate synthetic errors that are
    GENUINELY N(0, P_true) by construction (not from the EKF at all),
    and confirm the empirical mean NEES converges close to n=6 -- this
    validates compute_nees's math independent of any EKF machinery.
    """
    rng = np.random.default_rng(2024)
    n = 6
    P_true = np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])
    L = np.linalg.cholesky(P_true)

    n_samples = 5000
    nees_values = []
    for _ in range(n_samples):
        white_noise = rng.normal(size=n)
        e = L @ white_noise  # genuinely N(0, P_true) by construction
        nees_values.append(compute_nees(e, np.zeros(n), P_true))

    empirical_mean = np.mean(nees_values)
    # Expected mean is exactly n=6; 5000 samples gives a tight enough
    # empirical estimate that a 10% relative tolerance safely catches a
    # real implementation bug while avoiding sampling-noise flakiness.
    assert empirical_mean == pytest.approx(n, rel=0.10)


def test_consistency_interval_contains_theoretical_mean():
    """The confidence interval around a large sample count should straddle the theoretical mean."""
    lower, upper = consistency_interval(dof=6, n_samples=1000, alpha=0.05)
    assert lower < 6.0 < upper


def test_consistency_interval_narrows_with_more_samples():
    """More samples should produce a TIGHTER confidence interval around the mean -- basic sanity check."""
    lower_small, upper_small = consistency_interval(dof=6, n_samples=50, alpha=0.05)
    lower_large, upper_large = consistency_interval(dof=6, n_samples=5000, alpha=0.05)

    width_small = upper_small - lower_small
    width_large = upper_large - lower_large
    assert width_large < width_small


def _run_filter_and_collect_nees_nis(sigma_km: float, R_filter_scale: float, seed: int = 7):
    """
    Shared helper: run the EKF over the nominal scenario, collecting
    NEES and NIS at every step. R_filter_scale lets a test tell the
    filter the WRONG measurement noise level (used deliberately in the
    inconsistency test below).
    """
    rng = np.random.default_rng(seed)

    _, true_states = propagate_rk4(
        two_body_eom, t0=0.0, state0=STATE0, step_s=10.0, duration_s=3000.0, mu=MU_EARTH
    )
    measurements = generate_measurement_series(true_states, sigma_km=sigma_km, rng=rng)

    H = position_measurement_matrix()
    R_true = np.eye(3) * sigma_km**2
    R_filter = R_true * R_filter_scale  # what the FILTER believes, possibly wrong
    # Q tuned via a multi-seed sweep (see docs/mathematics.md) to bring
    # both NEES and NIS close to their theoretical means across 5
    # independent seeds -- NOT tuned against a single run, since a
    # single-seed tuning was tried first and found to be unrepresentative
    # (see the M5 development notes: seed 7 alone suggested a Q an order
    # of magnitude larger than what a 5-seed average actually supports).
    # Q here is larger than the "true" physical process noise (which is
    # ~zero for pure two-body dynamics) specifically to absorb EKF
    # linearization error -- a standard, documented tuning practice
    # (Bar-Shalom et al.), not a sign the dynamics model itself is wrong.
    Q = np.diag([3e-8, 3e-8, 3e-8, 3e-11, 3e-11, 3e-11])

    state = STATE0.copy()
    P = np.eye(6) * 1.0

    nees_values = []
    nis_values = []
    for k in range(1, len(true_states)):
        state, P = ekf_predict(state, P, dt=10.0, mu=MU_EARTH, Q=Q)
        state, P, innovation, S = ekf_update(state, P, measurements[k], H, R_filter)

        nees_values.append(compute_nees(true_states[k], state, P))
        nis_values.append(compute_nis(innovation, S))

    return np.array(nees_values), np.array(nis_values)


def test_correctly_tuned_filter_passes_consistency_check():
    """
    The REAL integration test: run the actual EKF from M4, with correctly-
    specified R (matching the true measurement noise), and confirm
    time-averaged NEES and NIS fall within their chi-squared confidence
    intervals -- proving this filter's reported uncertainty is honest,
    not just that its estimate is accurate.

    Averages over 5 independent seeds rather than a single run: a single
    trajectory's time-average is a documented approximation (see module
    docstring) with real sampling variance -- one unlucky seed can land
    just outside the interval even for a genuinely well-tuned filter.
    Averaging a handful of independent trials is a lightweight step
    toward the fully rigorous Monte Carlo approach (stretch goal S2),
    and correctly resolves this without cherry-picking a favorable seed.
    """
    nees_all = []
    nis_all = []
    for seed in [1, 2, 3, 4, 5]:
        nees_values, nis_values = _run_filter_and_collect_nees_nis(
            sigma_km=0.5, R_filter_scale=1.0, seed=seed
        )
        nees_all.append(nees_values[10:])
        nis_all.append(nis_values[10:])

    nees_steady = np.concatenate(nees_all)
    nis_steady = np.concatenate(nis_all)

    nees_lower, nees_upper = consistency_interval(dof=6, n_samples=len(nees_steady))
    nis_lower, nis_upper = consistency_interval(dof=3, n_samples=len(nis_steady))

    assert nees_lower < np.mean(nees_steady) < nees_upper, (
        f"NEES average {np.mean(nees_steady):.3f} outside consistency interval "
        f"[{nees_lower:.3f}, {nees_upper:.3f}]"
    )
    assert nis_lower < np.mean(nis_steady) < nis_upper, (
        f"NIS average {np.mean(nis_steady):.3f} outside consistency interval "
        f"[{nis_lower:.3f}, {nis_upper:.3f}]"
    )


def test_filter_with_wrong_measurement_noise_fails_consistency_check():
    """
    Proves the consistency test actually has teeth: deliberately tell
    the filter its measurements are 100x more precise than they truly
    are (R_filter_scale=0.01). The filter becomes overconfident -- its
    covariance shrinks far below the actual error -- so NIS should be
    well ABOVE the consistency interval's upper bound.
    """
    _, nis_values = _run_filter_and_collect_nees_nis(sigma_km=0.5, R_filter_scale=0.01)
    nis_steady = nis_values[10:]

    _, nis_upper = consistency_interval(dof=3, n_samples=len(nis_steady))

    assert np.mean(nis_steady) > nis_upper, (
        "Expected an overconfident (badly-tuned R) filter to FAIL the NIS "
        "consistency check by exceeding the upper bound -- it did not, "
        "meaning the consistency test may not be sensitive to real problems."
    )
