"""
Tests for the Extended Kalman Filter.

Split across genuinely different validation concerns:
    - Jacobian correctness (cross-checked against finite differences)
    - Predict step behavior (matches plain RK4; covariance grows)
    - Update step behavior (Kalman gain trust-weighting is physically correct)
    - Full filter loop (estimate tracks true trajectory over many cycles)
    - Numerical health (covariance stays symmetric/PSD under Joseph form)
"""

import numpy as np
import pytest

from orbitdet.dynamics.integrators import propagate_rk4
from orbitdet.dynamics.two_body import two_body_eom
from orbitdet.estimation.ekf import compute_dynamics_jacobian, ekf_predict, ekf_update
from orbitdet.measurements.models import (
    generate_measurement_series,
    position_measurement_matrix,
)

MU_EARTH = 398600.4418
R0 = np.array([6778.137, 0.0, 0.0])
V0 = np.array([0.0, 7.6685582, 0.0])
STATE0 = np.concatenate([R0, V0])


def _numerical_jacobian(state: np.ndarray, mu: float, eps: float = 1e-6) -> np.ndarray:
    """
    Finite-difference Jacobian of two_body_eom, used ONLY to
    cross-validate the analytical Jacobian -- same "trust but verify"
    principle as M1's RK4-vs-solve_ivp cross-check.
    """
    n = len(state)
    J = np.zeros((n, n))
    f0 = two_body_eom(0.0, state, mu)
    for i in range(n):
        perturbed = state.copy()
        perturbed[i] += eps
        f1 = two_body_eom(0.0, perturbed, mu)
        J[:, i] = (f1 - f0) / eps
    return J


def test_analytical_jacobian_matches_finite_difference():
    """The hand-derived gravity-gradient Jacobian must agree with a numerical cross-check."""
    A_analytical = compute_dynamics_jacobian(STATE0, MU_EARTH)
    A_numerical = _numerical_jacobian(STATE0, MU_EARTH)

    np.testing.assert_allclose(A_analytical, A_numerical, atol=1e-5)


def test_predict_step_state_matches_plain_rk4():
    """
    The predicted state (ignoring covariance) must match M1's plain
    propagate_rk4 exactly -- the EKF's state propagation is not a
    different, second implementation of the dynamics.
    """
    P0 = np.eye(6) * 1e-6
    Q = np.eye(6) * 1e-10

    state_pred, _ = ekf_predict(STATE0, P0, dt=10.0, mu=MU_EARTH, Q=Q)

    _, rk4_states = propagate_rk4(two_body_eom, t0=0.0, state0=STATE0, step_s=10.0, duration_s=10.0, mu=MU_EARTH)
    expected_state = rk4_states[-1]

    np.testing.assert_allclose(state_pred, expected_state, rtol=1e-10)


def test_predict_step_covariance_grows_without_measurements():
    """Uncertainty should increase during predict (no new information yet), never shrink."""
    P0 = np.eye(6) * 0.01
    Q = np.eye(6) * 1e-8

    _, P_pred = ekf_predict(STATE0, P0, dt=10.0, mu=MU_EARTH, Q=Q)

    assert np.trace(P_pred) > np.trace(P0)


def test_precise_measurement_pulls_estimate_close_to_measurement():
    """
    Kalman gain trust-weighting, direct test: with a VERY precise
    measurement (tiny R), the updated state should move close to the
    measurement itself, not stay near the (comparatively uncertain)
    prediction.
    """
    state_pred = STATE0.copy()
    P_pred = np.eye(6) * 1.0  # deliberately large prediction uncertainty

    H = position_measurement_matrix()
    true_position = STATE0[0:3]
    offset_measurement = true_position + np.array([5.0, 0.0, 0.0])  # measurement disagrees with prediction

    R_precise = np.eye(3) * 1e-8  # extremely trustworthy measurement

    state_updated, _, _, _ = ekf_update(state_pred, P_pred, offset_measurement, H, R_precise)

    assert np.linalg.norm(state_updated[0:3] - offset_measurement) < 0.01


def test_imprecise_measurement_barely_moves_estimate():
    """The opposite trust-weighting case: a very noisy measurement should barely shift the estimate."""
    state_pred = STATE0.copy()
    P_pred = np.eye(6) * 1e-6  # deliberately small (confident) prediction uncertainty

    H = position_measurement_matrix()
    true_position = STATE0[0:3]
    offset_measurement = true_position + np.array([5.0, 0.0, 0.0])

    R_imprecise = np.eye(3) * 1e6  # nearly useless measurement

    state_updated, _, _, _ = ekf_update(state_pred, P_pred, offset_measurement, H, R_imprecise)

    assert np.linalg.norm(state_updated[0:3] - state_pred[0:3]) < 0.01


def test_covariance_remains_symmetric_and_positive_semidefinite_over_many_cycles():
    """
    Numerical health check enabled specifically by the Joseph form
    update: run many predict/update cycles and confirm P never becomes
    asymmetric or develops a negative eigenvalue (unphysical).
    """
    rng = np.random.default_rng(2024)
    state = STATE0.copy()
    P = np.eye(6) * 0.1
    Q = np.diag([1e-10, 1e-10, 1e-10, 1e-12, 1e-12, 1e-12])
    R = np.eye(3) * 0.25
    H = position_measurement_matrix()

    true_state = STATE0.copy()
    for _ in range(200):
        true_state = propagate_rk4(two_body_eom, 0.0, true_state, 10.0, 10.0, MU_EARTH)[1][-1]
        state, P = ekf_predict(state, P, dt=10.0, mu=MU_EARTH, Q=Q)

        z = true_state[0:3] + rng.normal(0.0, 0.5, size=3)
        state, P, _, _ = ekf_update(state, P, z, H, R)

        assert np.allclose(P, P.T, atol=1e-9), "Covariance is no longer symmetric"
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues > -1e-9), "Covariance has a negative eigenvalue (not PSD)"


def test_filter_estimate_tracks_true_trajectory_over_full_scenario():
    """
    The actual integration test: run the full predict/update loop over
    a realistic scenario and confirm the filter's position error stays
    bounded, at a scale consistent with the measurement noise -- this
    is "does the filter actually work," not just "do the equations run."
    """
    rng = np.random.default_rng(7)
    sigma_km = 0.5

    _, true_states = propagate_rk4(two_body_eom, t0=0.0, state0=STATE0, step_s=10.0, duration_s=3000.0, mu=MU_EARTH)
    measurements = generate_measurement_series(true_states, sigma_km=sigma_km, rng=rng)

    H = position_measurement_matrix()
    R = np.eye(3) * sigma_km**2
    Q = np.diag([1e-10, 1e-10, 1e-10, 1e-12, 1e-12, 1e-12])

    # Deliberately imperfect initial guess -- the filter must converge
    # toward truth despite not starting there.
    state = STATE0 + np.array([2.0, -2.0, 1.0, 0.001, -0.001, 0.0])
    P = np.eye(6) * 4.0

    position_errors = []
    for k in range(1, len(true_states)):
        state, P = ekf_predict(state, P, dt=10.0, mu=MU_EARTH, Q=Q)
        state, P, _, _ = ekf_update(state, P, measurements[k], H, R)
        position_errors.append(np.linalg.norm(state[0:3] - true_states[k, 0:3]))

    # After the initial transient, steady-state error should be
    # comparable to the measurement noise level, not diverging.
    steady_state_error = np.mean(position_errors[-50:])
    assert steady_state_error < 3 * sigma_km, (
        f"Steady-state position error {steady_state_error:.3f} km is too large "
        f"relative to measurement sigma {sigma_km} km -- filter may be diverging."
    )
