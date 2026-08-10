"""
Two-body dynamics: the equation of motion and its conserved quantities.

Derivation (see docs/mathematics.md for the full write-up):

    Two point masses under mutual gravitation reduce EXACTLY to a single
    relative-motion equation:

        r_ddot = -mu * r / |r|^3

    where r is the position of the spacecraft relative to the central
    body, and mu = G*(M_central + m_spacecraft) approx= G*M_central,
    since a real spacecraft's mass is utterly negligible next to Earth's.

Conserved quantities (both proven by direct differentiation, see
docs/mathematics.md):
    - specific orbital energy:      epsilon = v^2/2 - mu/r
    - specific angular momentum:    h = r cross v  (a full VECTOR, not
      just its magnitude -- constant direction is what confines the
      orbit to a fixed plane)

These are the primary validation tools for M1: a numerically-integrated
two-body orbit MUST conserve both to within numerical tolerance. If it
doesn't, the integrator or the EOM implementation has a bug.
"""

from __future__ import annotations

import numpy as np


def two_body_eom(t: float, state: np.ndarray, mu: float) -> np.ndarray:
    """
    Two-body equation of motion, in first-order state-space form.

    Signature matches scipy.integrate.solve_ivp's expected f(t, y, *args)
    exactly, so this SAME function is used both by our hand-written RK4
    propagator and by solve_ivp during cross-validation -- there is only
    ever one implementation of the physics to get wrong.

    Args:
        t: elapsed time, seconds. Unused -- two-body dynamics under the
           Keplerian assumption have no explicit time dependence (the
           system is "autonomous"), but the argument is kept so this
           function's signature matches what generic ODE solvers expect.
        state: (6,) array [rx, ry, rz, vx, vy, vz], km and km/s.
        mu: gravitational parameter, km^3/s^2.

    Returns:
        (6,) array [vx, vy, vz, ax, ay, az] -- the time derivative of state.
    """
    r = state[0:3]
    v = state[3:6]

    r_mag = np.linalg.norm(r)
    a = -mu * r / r_mag**3

    return np.concatenate([v, a])


def specific_orbital_energy(state: np.ndarray, mu: float) -> float:
    """
    Specific (per unit mass) orbital energy, km^2/s^2.

    epsilon = v^2/2 - mu/r

    Conserved exactly under two-body dynamics. Negative for a bound
    (elliptical) orbit, zero for exactly parabolic escape, positive for
    a hyperbolic (escaping) trajectory.
    """
    r = state[0:3]
    v = state[3:6]
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    return v_mag**2 / 2.0 - mu / r_mag


def specific_angular_momentum(state: np.ndarray) -> np.ndarray:
    """
    Specific (per unit mass) angular momentum VECTOR, km^2/s.

    h = r cross v

    Conserved exactly under two-body dynamics -- both magnitude AND
    direction. Constant direction is what confines the orbit to a fixed
    plane (h is always perpendicular to r, so r sweeps out the plane
    perpendicular to h).
    """
    r = state[0:3]
    v = state[3:6]
    return np.cross(r, v)
