"""
Tests for statistical anomaly detection.

Covers: threshold correctness, single-sample flagging, the false-alarm
rate matching alpha under genuinely nominal conditions (a statistical
validation, same spirit as M3/M5), persistence-based false-alarm
suppression, and detection of a REAL injected orbital perturbation
(directly analogous to Mission Control's anomaly injector, but detected
via principled chi-squared hypothesis testing rather than a fixed
numeric threshold).
"""

import numpy as np
import pytest
from scipy.stats import chi2

from orbitdet.detection.anomaly import (
    detect_with_persistence,
    is_single_sample_anomalous,
    nis_threshold,
)
from orbitdet.dynamics.integrators import propagate_rk4
from orbitdet.dynamics.two_body import two_body_eom
from orbitdet.estimation.consistency import compute_nis
from orbitdet.estimation.ekf import ekf_predict, ekf_update
from orbitdet.measurements.models import generate_measurement_series, position_measurement_matrix

MU_EARTH = 398600.4418
R0 = np.array([6778.137, 0.0, 0.0])
V0 = np.array([0.0, 7.6685582, 0.0])
STATE0 = np.concatenate([R0, V0])


def test_nis_threshold_matches_scipy_chi2_quantile_directly():
    """Sanity check against scipy's own chi2 quantile function, computed independently."""
    threshold = nis_threshold(dof=3, alpha=0.01)
    expected = chi2.ppf(0.99, df=3)
    assert threshold == pytest.approx(expected)


def test_single_sample_flag_boundary_behavior():
    threshold = nis_threshold(dof=3, alpha=0.01)
    assert is_single_sample_anomalous(threshold + 0.01, threshold) is True
    assert is_single_sample_anomalous(threshold - 0.01, threshold) is False


def test_false_alarm_rate_matches_alpha_under_nominal_chi_squared_samples():
    """
    Statistical validation: generate GENUINE chi-squared(3) samples (not
    from the EKF -- isolated from any filter machinery) and confirm the
    empirical single-sample false-alarm rate is close to the configured
    alpha. This validates the threshold's statistical meaning directly.
    """
    rng = np.random.default_rng(99)
    alpha = 0.01
    threshold = nis_threshold(dof=3, alpha=alpha)

    n_samples = 20000
    samples = rng.chisquare(df=3, size=n_samples)
    empirical_false_alarm_rate = np.mean(samples > threshold)

    # alpha=0.01 with 20000 samples: expected false alarms ~200, with
    # sampling std ~sqrt(20000*0.01*0.99)~14 -- a relative tolerance of
    # 30% comfortably avoids flakiness while still catching a real
    # threshold-computation bug (e.g. wrong dof, or 1-alpha vs alpha flip).
    assert empirical_false_alarm_rate == pytest.approx(alpha, rel=0.30)


def test_persistence_detector_ignores_isolated_single_sample_spike():
    """
    A single isolated high value (an expected, occasional false alarm)
    surrounded by nominal values should NOT trigger a confirmed
    detection under M-of-N persistence logic.
    """
    threshold = 10.0
    nis_series = np.array([2.0, 3.0, 15.0, 2.5, 3.0, 2.0, 3.5, 2.0])  # one isolated spike at index 2

    confirmed = detect_with_persistence(nis_series, threshold, required_hits=3, window=5)

    assert not np.any(confirmed), "An isolated single-sample spike should not confirm a persistent anomaly"


def test_persistence_detector_confirms_sustained_deviation():
    """A sustained run of high values should be confirmed by persistence logic, unlike an isolated spike."""
    threshold = 10.0
    nis_series = np.array([2.0, 2.5, 15.0, 16.0, 14.0, 17.0, 15.0, 2.0, 2.5])

    confirmed = detect_with_persistence(nis_series, threshold, required_hits=3, window=5)

    assert np.any(confirmed), "A sustained deviation should be confirmed by persistence logic"
    # Confirmation should occur somewhere within the sustained high region (indices 2-6).
    assert np.any(confirmed[2:7])


def test_detects_real_injected_velocity_perturbation():
    """
    The actual integration test: run the EKF over a scenario where, at
    a known step, we inject a sudden velocity change directly into the
    TRUE trajectory (an unmodeled, unplanned perturbation the filter
    has no knowledge of) -- directly analogous to Mission Control's
    anomaly injector, but detected here via principled chi-squared
    hypothesis testing rather than a fixed numeric threshold.

    Confirms: no confirmed detection before the injection, and a
    confirmed detection within a small number of steps after it.
    """
    rng = np.random.default_rng(42)
    sigma_km = 0.5
    injection_step = 150
    dt = 10.0

    # Build the true trajectory in two pieces: nominal up to the
    # injection point, then a sudden 0.5 km/s velocity kick (a
    # significant unplanned Delta-v) applied directly to truth, with
    # nominal two-body propagation continuing afterward from the new,
    # perturbed state. The FILTER never sees this kick directly -- it
    # only sees its effect through subsequent noisy measurements.
    _, true_states_before = propagate_rk4(
        two_body_eom, t0=0.0, state0=STATE0, step_s=dt,
        duration_s=injection_step * dt, mu=MU_EARTH,
    )
    perturbed_state = true_states_before[-1].copy()
    perturbed_state[3:6] += np.array([0.5, 0.0, 0.0])  # velocity kick, km/s

    _, true_states_after = propagate_rk4(
        two_body_eom, t0=0.0, state0=perturbed_state, step_s=dt,
        duration_s=100 * dt, mu=MU_EARTH,
    )
    true_states = np.vstack([true_states_before, true_states_after[1:]])

    measurements = generate_measurement_series(true_states, sigma_km=sigma_km, rng=rng)

    H = position_measurement_matrix()
    R = np.eye(3) * sigma_km**2
    Q = np.diag([3e-8, 3e-8, 3e-8, 3e-11, 3e-11, 3e-11])  # same tuned Q validated in M5

    state = STATE0.copy()
    P = np.eye(6) * 1.0

    nis_series = []
    for k in range(1, len(true_states)):
        state, P = ekf_predict(state, P, dt=dt, mu=MU_EARTH, Q=Q)
        state, P, innovation, S = ekf_update(state, P, measurements[k], H, R)
        nis_series.append(compute_nis(innovation, S))
    nis_series = np.array(nis_series)

    threshold = nis_threshold(dof=3, alpha=0.01)
    confirmed = detect_with_persistence(nis_series, threshold, required_hits=3, window=5)

    # nis_series index k-1 corresponds to true_states index k (loop starts at k=1).
    injection_index_in_series = injection_step - 1

    pre_injection_confirmed = confirmed[: injection_index_in_series - 5]  # margin before injection
    assert not np.any(pre_injection_confirmed), (
        "Confirmed anomaly detected BEFORE the injected perturbation occurred -- "
        "false positive during a genuinely nominal period."
    )

    post_injection_window = confirmed[injection_index_in_series : injection_index_in_series + 20]
    assert np.any(post_injection_window), (
        "Failed to confirm the injected velocity perturbation within 20 steps of it occurring."
    )
