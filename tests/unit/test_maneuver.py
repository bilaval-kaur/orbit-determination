"""
Tests for the impulsive maneuver model.

Includes a direct test of a provable orbital mechanics fact: a
tangential burn changes specific orbital energy at FIRST order in
delta_v, while a normal (cross-track) burn of equal magnitude changes
it only at SECOND order -- this is why plane-change maneuvers are
famously expensive and along-track burns are the efficient way to
resize an orbit.
"""

import numpy as np
import pytest

from orbitdet.dynamics.two_body import specific_orbital_energy
from orbitdet.maneuver.impulsive import (
    apply_impulsive_maneuver,
    apply_maneuver_and_propagate,
    delta_v_rtn_to_eci,
    maneuver_delta_v_cost,
    rtn_to_eci_matrix,
)

MU_EARTH = 398600.4418
R0 = np.array([6778.137, 0.0, 0.0])
V0 = np.array([0.0, 7.6685582, 0.0])
STATE0 = np.concatenate([R0, V0])


def test_rtn_frame_is_orthonormal_and_right_handed():
    """R_hat, T_hat, N_hat must form a proper orthonormal, right-handed basis."""
    rotation = rtn_to_eci_matrix(STATE0)
    R_hat, T_hat, N_hat = rotation[:, 0], rotation[:, 1], rotation[:, 2]

    assert np.linalg.norm(R_hat) == pytest.approx(1.0)
    assert np.linalg.norm(T_hat) == pytest.approx(1.0)
    assert np.linalg.norm(N_hat) == pytest.approx(1.0)

    assert np.dot(R_hat, T_hat) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(R_hat, N_hat) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(T_hat, N_hat) == pytest.approx(0.0, abs=1e-10)

    np.testing.assert_allclose(np.cross(R_hat, T_hat), N_hat, atol=1e-10)


def test_maneuver_leaves_position_unchanged():
    """The impulsive assumption: position must be EXACTLY unchanged by a burn, only velocity jumps."""
    delta_v = np.array([0.1, 0.0, 0.0])
    new_state = apply_impulsive_maneuver(STATE0, delta_v)

    np.testing.assert_array_equal(new_state[0:3], STATE0[0:3])
    assert not np.array_equal(new_state[3:6], STATE0[3:6])


def test_rtn_to_eci_conversion_preserves_magnitude():
    """Rotation is orthogonal -- it must preserve vector length exactly."""
    delta_v_rtn = np.array([0.05, 0.2, -0.03])
    delta_v_eci = delta_v_rtn_to_eci(STATE0, delta_v_rtn)

    assert np.linalg.norm(delta_v_eci) == pytest.approx(np.linalg.norm(delta_v_rtn))


def test_maneuver_cost_equals_delta_v_magnitude():
    delta_v = np.array([0.3, 0.4, 0.0])
    assert maneuver_delta_v_cost(delta_v) == pytest.approx(0.5)  # 3-4-5 triangle


def test_prograde_burn_increases_orbital_energy():
    """A positive along-track (T) burn should raise specific orbital energy (raises the orbit)."""
    prograde_dv = np.array([0.0, 0.1, 0.0])  # RTN: pure +T
    delta_v_eci = delta_v_rtn_to_eci(STATE0, prograde_dv)
    new_state = apply_impulsive_maneuver(STATE0, delta_v_eci)

    energy_before = specific_orbital_energy(STATE0, MU_EARTH)
    energy_after = specific_orbital_energy(new_state, MU_EARTH)

    assert energy_after > energy_before


def test_retrograde_burn_decreases_orbital_energy():
    """A negative along-track (T) burn should lower specific orbital energy (lowers the orbit)."""
    retrograde_dv = np.array([0.0, -0.1, 0.0])
    delta_v_eci = delta_v_rtn_to_eci(STATE0, retrograde_dv)
    new_state = apply_impulsive_maneuver(STATE0, delta_v_eci)

    energy_before = specific_orbital_energy(STATE0, MU_EARTH)
    energy_after = specific_orbital_energy(new_state, MU_EARTH)

    assert energy_after < energy_before


def test_tangential_burn_changes_energy_far_more_than_normal_burn_of_equal_magnitude():
    """
    The provable, testable fact from docs/mathematics.md: for the SAME
    small delta_v magnitude, a tangential burn changes specific orbital
    energy at first order (change proportional to delta_v), while a
    normal burn changes it only at second order (proportional to
    delta_v^2) -- so for small delta_v, the tangential energy change
    must be MUCH larger than the normal one. This is the rigorous basis
    for "plane changes are expensive."
    """
    dv_magnitude = 0.01  # small, to clearly separate first- vs second-order effects

    tangential_dv = np.array([0.0, dv_magnitude, 0.0])
    normal_dv = np.array([0.0, 0.0, dv_magnitude])

    energy_before = specific_orbital_energy(STATE0, MU_EARTH)

    state_after_tangential = apply_impulsive_maneuver(
        STATE0, delta_v_rtn_to_eci(STATE0, tangential_dv)
    )
    state_after_normal = apply_impulsive_maneuver(
        STATE0, delta_v_rtn_to_eci(STATE0, normal_dv)
    )

    energy_change_tangential = abs(specific_orbital_energy(state_after_tangential, MU_EARTH) - energy_before)
    energy_change_normal = abs(specific_orbital_energy(state_after_normal, MU_EARTH) - energy_before)

    # Same delta_v magnitude, but tangential should dominate by roughly
    # a factor of (orbital velocity / dv_magnitude) -- a large ratio.
    # Requiring at least 10x is a conservative, clearly-first-vs-second-
    # order-distinguishing margin, not a tight numerical coincidence.
    assert energy_change_tangential > 10 * energy_change_normal


def test_apply_maneuver_and_propagate_produces_valid_trajectory():
    """Integration check: the full burn + propagate pipeline runs and returns a sensible trajectory."""
    prograde_dv = np.array([0.0, 0.05, 0.0])
    times, states = apply_maneuver_and_propagate(
        STATE0, prograde_dv, propagate_duration_s=1000.0, step_s=10.0, mu=MU_EARTH
    )

    assert states.shape[1] == 6
    assert len(times) == len(states)
    # Post-burn orbit should still be a valid bound orbit (negative energy).
    assert specific_orbital_energy(states[0], MU_EARTH) < 0
