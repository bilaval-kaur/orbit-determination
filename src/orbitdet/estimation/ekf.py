"""
Extended Kalman Filter for two-body orbit state estimation.

See docs/mathematics.md for the full derivation. Summary of the two-step
cycle:

    PREDICT:
        x_pred = f(x_prev)                  -- propagate state (M1's RK4)
        P_pred = Phi @ P_prev @ Phi.T + Q   -- propagate covariance

    UPDATE (given measurement z):
        innovation = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ inv(S)
        x_new = x_pred + K @ innovation
        P_new = (I - K@H) @ P_pred @ (I - K@H).T + K @ R @ K.T   (Joseph form)

Phi (the state transition matrix) is propagated by integrating the
"variational equations" Phi_dot = A(x) @ Phi ALONGSIDE the state, using
RK4 -- a direct extension of M1's integrator to a coupled, augmented
system, not a new integration method.
"""

from __future__ import annotations

import numpy as np

from orbitdet.dynamics.two_body import two_body_eom


def compute_dynamics_jacobian(state: np.ndarray, mu: float) -> np.ndarray:
    """
    Analytical Jacobian A = df/dx of the two-body equation of motion.

    Block structure:
        A = [ 0_3   I_3 ]
            [ G     0_3 ]

    where G = -mu/r^3 * (I_3 - 3 * r_hat @ r_hat.T) is the gravity-
    gradient ("tidal") matrix -- derived in docs/mathematics.md using
    the same partial-derivative technique as M1's energy conservation
    proof, applied to acceleration instead of energy.
    """
    r = state[0:3]
    r_mag = np.linalg.norm(r)
    r_hat = r / r_mag

    G = -mu / r_mag**3 * (np.eye(3) - 3.0 * np.outer(r_hat, r_hat))

    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = G
    return A


def _augmented_derivative(
    state: np.ndarray, Phi: np.ndarray, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derivative of the coupled [state, Phi] system:
        x_dot   = f(x)
        Phi_dot = A(x) @ Phi
    """
    x_dot = two_body_eom(0.0, state, mu)
    A = compute_dynamics_jacobian(state, mu)
    Phi_dot = A @ Phi
    return x_dot, Phi_dot


def _rk4_step_with_stm(
    state: np.ndarray, Phi: np.ndarray, h: float, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    One RK4 step advancing state and state transition matrix together.

    This is M1's rk4_step, extended to the augmented [state, Phi]
    system -- the four k-stages are evaluated at CONSISTENT (state, Phi)
    pairs, since Phi's dynamics depend on the instantaneous state.
    """
    k1_x, k1_p = _augmented_derivative(state, Phi, mu)
    k2_x, k2_p = _augmented_derivative(state + h / 2 * k1_x, Phi + h / 2 * k1_p, mu)
    k3_x, k3_p = _augmented_derivative(state + h / 2 * k2_x, Phi + h / 2 * k2_p, mu)
    k4_x, k4_p = _augmented_derivative(state + h * k3_x, Phi + h * k3_p, mu)

    new_state = state + (h / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    new_Phi = Phi + (h / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
    return new_state, new_Phi


def ekf_predict(
    state: np.ndarray, P: np.ndarray, dt: float, mu: float, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF predict (time update) step.

    Scope note: propagates over exactly one step of duration dt,
    matching the measurement cadence -- not multiple integrator
    substeps between sparser measurements. A reasonable simplification
    for this project's scope; generalizing to multiple substeps per
    predict call is straightforward if a future experiment needs it.

    Args:
        state: (6,) current state estimate.
        P: (6,6) current covariance estimate.
        dt: time to propagate forward, seconds.
        mu: gravitational parameter, km^3/s^2.
        Q: (6,6) process noise covariance.

    Returns:
        (state_pred, P_pred)
    """
    Phi0 = np.eye(6)
    state_pred, Phi = _rk4_step_with_stm(state, Phi0, dt, mu)
    P_pred = Phi @ P @ Phi.T + Q
    return state_pred, P_pred


def ekf_update(
    state_pred: np.ndarray,
    P_pred: np.ndarray,
    measurement: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    EKF update (measurement update) step.

    Args:
        state_pred: (6,) predicted state (from ekf_predict).
        P_pred: (6,6) predicted covariance (from ekf_predict).
        measurement: (m,) actual measurement.
        H: (m,6) measurement matrix.
        R: (m,m) measurement noise covariance.

    Returns:
        (state_updated, P_updated, innovation, S) -- innovation and S
        are returned (not just discarded) because M5's NEES/NIS
        consistency tests need them directly.
    """
    innovation = measurement - H @ state_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    state_updated = state_pred + K @ innovation

    I = np.eye(P_pred.shape[0])
    A = I - K @ H
    # Joseph form: numerically robust, guarantees P stays symmetric and
    # positive semi-definite even under floating-point roundoff
    # accumulated over many filter iterations -- unlike the simpler
    # (I - K@H) @ P_pred form, which can drift asymmetric over time.
    P_updated = A @ P_pred @ A.T + K @ R @ K.T

    return state_updated, P_updated, innovation, S
