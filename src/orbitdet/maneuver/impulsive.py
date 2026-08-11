"""
Impulsive maneuver model: applies an instantaneous velocity change to a
state, expressed in the RTN (Radial-Transverse-Normal) frame -- the
physically intuitive, spacecraft-attached frame real mission planners
use to describe burns.

Impulsive assumption: burn duration << orbital period, so the burn is
modeled as position-continuous, velocity-discontinuous:
    r(t_burn+) = r(t_burn-)
    v(t_burn+) = v(t_burn-) + delta_v
No new dynamics are needed -- after the burn, the state is simply a new
initial condition for M1's existing two-body RK4 propagator.

RTN frame, defined at a given state (r, v):
    R_hat = r / |r|                     (radial, outward)
    N_hat = (r x v) / |r x v|           (normal, along angular momentum)
    T_hat = N_hat x R_hat                (transverse, completes right-hand frame)

See docs/mathematics.md for the derivation of why a TANGENTIAL burn
changes specific orbital energy at FIRST order in delta_v, while a
NORMAL burn changes it only at SECOND order -- this is the rigorous
basis for "tangential burns are the efficient way to resize an orbit;
plane changes (normal burns) are expensive."
"""

from __future__ import annotations

import numpy as np


def rtn_to_eci_matrix(state: np.ndarray) -> np.ndarray:
    """
    Rotation matrix from the RTN frame to ECI, at the given state.

    Columns are [R_hat, T_hat, N_hat], each expressed in ECI. A vector
    given in RTN coordinates is converted to ECI via matrix @ vector_rtn.
    """
    r = state[0:3]
    v = state[3:6]

    R_hat = r / np.linalg.norm(r)

    h = np.cross(r, v)
    N_hat = h / np.linalg.norm(h)

    T_hat = np.cross(N_hat, R_hat)

    return np.column_stack([R_hat, T_hat, N_hat])


def delta_v_rtn_to_eci(state: np.ndarray, delta_v_rtn: np.ndarray) -> np.ndarray:
    """
    Convert a Delta-v specified in RTN components [dv_R, dv_T, dv_N]
    (km/s) to ECI components, at the given state.
    """
    rotation = rtn_to_eci_matrix(state)
    return rotation @ delta_v_rtn


def apply_impulsive_maneuver(state: np.ndarray, delta_v_eci: np.ndarray) -> np.ndarray:
    """
    Apply an impulsive maneuver: position unchanged, velocity jumps by
    delta_v_eci (already expressed in ECI, km/s).
    """
    new_state = state.copy()
    new_state[3:6] = new_state[3:6] + delta_v_eci
    return new_state


def maneuver_delta_v_cost(delta_v_eci: np.ndarray) -> float:
    """
    The maneuver's cost, in this project's currency: Delta-v magnitude,
    km/s -- NOT propellant mass. Per docs/conventions.md Section 5, mass
    never enters this project's units; converting to propellant mass via
    the rocket equation is an explicitly out-of-scope future extension.
    """
    return float(np.linalg.norm(delta_v_eci))


def apply_maneuver_and_propagate(
    state: np.ndarray,
    delta_v_rtn: np.ndarray,
    propagate_duration_s: float,
    step_s: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: apply an RTN-specified burn, then propagate the
    resulting orbit forward using M1's existing RK4 propagator. Used
    directly by M8's maneuver evaluation and optimization.

    Returns:
        (times, states) from propagate_rk4, starting immediately after
        the burn (t=0 at the post-burn state).
    """
    from orbitdet.dynamics.integrators import propagate_rk4
    from orbitdet.dynamics.two_body import two_body_eom

    delta_v_eci = delta_v_rtn_to_eci(state, delta_v_rtn)
    post_burn_state = apply_impulsive_maneuver(state, delta_v_eci)

    return propagate_rk4(
        two_body_eom, t0=0.0, state0=post_burn_state,
        step_s=step_s, duration_s=propagate_duration_s, mu=mu,
    )
