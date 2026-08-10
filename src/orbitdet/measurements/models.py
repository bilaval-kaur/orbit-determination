"""
Measurement simulation: generates noisy observations of the true
spacecraft state, standing in for an imperfect onboard sensor (e.g. a
GPS-like navigation receiver reporting a position fix).

Model: z = h(x_true) + v, where v ~ N(0, R).

Scope (see M0 planning discussion): direct position measurement only,
h(x) = H @ x with H = [I_3 | 0_3] -- a LINEAR measurement function, so
no Jacobian is needed for this measurement type. Range/range-rate
(nonlinear h, real Jacobian) is stretch goal S3, deliberately deferred.

The noise model here is Gaussian by necessity, not convenience: the
Kalman filter (M4) is only provably optimal under a Gaussian noise
assumption, and the filter consistency tests (M5, NEES/NIS) are built
on that same assumption. The simulated noise and the filter's assumed
R must describe the same distribution, or "consistency" is meaningless.

Per docs/conventions.md Section 6, all randomness here MUST be drawn
from a caller-supplied, explicitly-seeded numpy.random.Generator --
never from NumPy's legacy global random state -- so that two runs of
the same scenario config produce bit-identical measurements.
"""

from __future__ import annotations

import numpy as np


def position_measurement_matrix() -> np.ndarray:
    """
    The (3, 6) measurement matrix H for direct position measurement.

    h(x) = H @ x extracts just the position components [rx, ry, rz]
    from the full 6-element state -- possible as a simple constant
    matrix (rather than a general nonlinear function needing a
    Jacobian) only because docs/conventions.md fixes the state vector
    ordering as position-then-velocity.
    """
    H = np.zeros((3, 6))
    H[0:3, 0:3] = np.eye(3)
    return H


def simulate_position_measurement(
    true_state: np.ndarray, sigma_km: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate one noisy position measurement from a true state.

    Args:
        true_state: (6,) true state vector [r, v], km and km/s.
        sigma_km: standard deviation of the (isotropic, i.e. identical
            in all three axes) Gaussian measurement noise, km. Isotropic
            is a simplifying starting assumption -- a real sensor's
            error may differ by axis; see Future Improvements.
        rng: an explicitly-seeded numpy.random.Generator (never the
            legacy global random state -- see docs/conventions.md).

    Returns:
        (3,) noisy position measurement, km.
    """
    H = position_measurement_matrix()
    true_position = H @ true_state

    noise = rng.normal(loc=0.0, scale=sigma_km, size=3)
    return true_position + noise


def generate_measurement_series(
    true_states: np.ndarray, sigma_km: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate noisy position measurements for an entire trajectory at once.

    Args:
        true_states: (N, 6) array of true states over time (e.g. the
            output of propagate_rk4 from M1).
        sigma_km: measurement noise standard deviation, km.
        rng: an explicitly-seeded numpy.random.Generator.

    Returns:
        (N, 3) array of noisy position measurements.
    """
    n = true_states.shape[0]
    H = position_measurement_matrix()
    true_positions = true_states @ H.T  # (N, 6) @ (6, 3) -> (N, 3)

    noise = rng.normal(loc=0.0, scale=sigma_km, size=(n, 3))
    return true_positions + noise
