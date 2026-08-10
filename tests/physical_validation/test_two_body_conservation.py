"""
Physical validation tests for two-body dynamics + RK4 propagation.

These are PHYSICAL VALIDATION tests (distinct from plain unit tests):
they check that the integrated system obeys real physics -- conservation
laws that are analytically provable for the two-body problem -- rather
than just checking that a function returns some expected value.

Uses the nominal circular LEO scenario (configs/nominal_two_body.yaml)
as the standard validation case: a circular orbit has trivial, exactly-
known analytical behavior (constant radius, constant speed, known
period), which makes it an unusually good sanity check.
"""

import math

import numpy as np
import pytest

from orbitdet.dynamics.integrators import propagate_rk4
from orbitdet.dynamics.two_body import (
    specific_angular_momentum,
    specific_orbital_energy,
    two_body_eom,
)

MU_EARTH = 398600.4418

R0 = np.array([6778.137, 0.0, 0.0])
V0 = np.array([0.0, 7.6685582, 0.0])
STATE0 = np.concatenate([R0, V0])


def test_specific_energy_is_conserved_over_multiple_orbits():
    """
    Propagate for ~3.2 orbital periods and confirm specific orbital
    energy stays constant to within tight relative tolerance. A
    meaningful drift here means the EOM or the integrator has a bug --
    energy conservation is not optional for two-body motion.
    """
    times, states = propagate_rk4(
        two_body_eom, t0=0.0, state0=STATE0, step_s=10.0, duration_s=18000.0, mu=MU_EARTH
    )

    energies = np.array([specific_orbital_energy(s, MU_EARTH) for s in states])
    initial_energy = energies[0]

    max_relative_drift = np.max(np.abs((energies - initial_energy) / initial_energy))
    assert max_relative_drift < 1e-9, (
        f"Specific energy drifted by relative {max_relative_drift:.2e} -- "
        "expected < 1e-9 for a correct two-body RK4 propagation at this step size."
    )


def test_specific_angular_momentum_vector_is_conserved():
    """
    Angular momentum must be conserved as a full VECTOR (magnitude AND
    direction) -- constant direction is what keeps the orbit planar.
    Checking only the magnitude would miss a bug that rotated the orbit
    plane while preserving |h|.
    """
    times, states = propagate_rk4(
        two_body_eom, t0=0.0, state0=STATE0, step_s=10.0, duration_s=18000.0, mu=MU_EARTH
    )

    h_vectors = np.array([specific_angular_momentum(s) for s in states])
    h0 = h_vectors[0]

    max_relative_drift = np.max(
        np.linalg.norm(h_vectors - h0, axis=1) / np.linalg.norm(h0)
    )
    assert max_relative_drift < 1e-9, (
        f"Angular momentum vector drifted by relative {max_relative_drift:.2e} -- "
        "orbit plane should be exactly fixed for two-body motion."
    )


def test_circular_orbit_returns_to_start_after_one_analytical_period():
    """
    Independent cross-check beyond conservation laws: a circular orbit's
    period is known in exact closed form, T = 2*pi*sqrt(r^3/mu). After
    propagating for exactly one such period, position should return
    close to the starting point -- this catches bugs that could
    theoretically still conserve energy/momentum (e.g. a consistently
    mis-scaled force) but would NOT reproduce the correct period.
    """
    r0_mag = np.linalg.norm(R0)
    period_s = 2 * math.pi * math.sqrt(r0_mag**3 / MU_EARTH)

    times, states = propagate_rk4(
        two_body_eom, t0=0.0, state0=STATE0, step_s=10.0, duration_s=period_s, mu=MU_EARTH
    )

    final_position = states[-1, 0:3]
    position_error_km = np.linalg.norm(final_position - R0)

    # Tolerance: the step size (10s) doesn't divide the analytical period
    # evenly, so the LAST step overshoots the true period slightly --
    # this tolerance accounts for that partial-step discretization, not
    # integrator error, which is far smaller (see the energy test above).
    assert position_error_km < 50.0, (
        f"Position error after one period was {position_error_km:.3f} km -- "
        "expected the spacecraft to return close to its starting point."
    )


def test_orbit_stays_at_constant_radius_for_circular_case():
    """
    A circular orbit's radius (distance from central body) is constant
    by definition. Checking this directly, in addition to the more
    abstract energy/momentum checks, gives an intuitive, easy-to-debug
    signal if something is wrong.
    """
    times, states = propagate_rk4(
        two_body_eom, t0=0.0, state0=STATE0, step_s=10.0, duration_s=18000.0, mu=MU_EARTH
    )

    radii = np.linalg.norm(states[:, 0:3], axis=1)
    initial_radius = radii[0]

    max_relative_deviation = np.max(np.abs(radii - initial_radius) / initial_radius)
    # Tolerance justification: RK4 at a 10s step over ~555 steps accumulates
    # floating-point and O(h^4) truncation noise at roughly the 1e-8 relative
    # level -- this corresponds to a physical drift of well under a meter on
    # a ~6800 km orbit. 5e-8 gives headroom above that observed noise floor
    # while remaining tight enough to catch a real bug, which would produce
    # drift orders of magnitude larger.
    assert max_relative_deviation < 5e-8
