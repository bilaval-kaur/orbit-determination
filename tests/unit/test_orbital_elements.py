"""
Tests for orbital elements <-> state vector conversion.

Uses a non-degenerate scenario (distinct nonzero eccentricity and
inclination) as the primary validation case -- the M1 nominal scenario
is circular AND equatorial, which is exactly the double-singularity
case this module has to handle specially, and therefore a poor choice
for validating the GENERAL conversion algorithm.
"""

import math

import numpy as np
import pytest

from orbitdet.dynamics.two_body import specific_orbital_energy
from orbitdet.elements.conversions import (
    OrbitalElements,
    elements_to_state,
    state_to_elements,
)

MU_EARTH = 398600.4418

# A deliberately non-degenerate GTO-like orbit: every element has a
# distinct, well-defined, nonzero value, so every branch of the
# conversion algorithm (RAAN quadrant check, arg-periapsis quadrant
# check, true-anomaly quadrant check) actually gets exercised.
NONDEGENERATE_ELEMENTS = OrbitalElements(
    semi_major_axis_km=24000.0,
    eccentricity=0.7,
    inclination_rad=math.radians(28.5),
    raan_rad=math.radians(45.0),
    arg_periapsis_rad=math.radians(30.0),
    true_anomaly_rad=math.radians(40.0),
    is_circular=False,
    is_equatorial=False,
)


def test_elements_to_state_to_elements_roundtrip():
    """
    Converting elements -> state -> elements should recover the
    original elements to tight tolerance, for a well-conditioned
    (non-degenerate) orbit.
    """
    state = elements_to_state(NONDEGENERATE_ELEMENTS, MU_EARTH)
    recovered = state_to_elements(state, MU_EARTH)

    assert recovered.semi_major_axis_km == pytest.approx(NONDEGENERATE_ELEMENTS.semi_major_axis_km, rel=1e-9)
    assert recovered.eccentricity == pytest.approx(NONDEGENERATE_ELEMENTS.eccentricity, rel=1e-9)
    assert recovered.inclination_rad == pytest.approx(NONDEGENERATE_ELEMENTS.inclination_rad, rel=1e-9)
    assert recovered.raan_rad == pytest.approx(NONDEGENERATE_ELEMENTS.raan_rad, rel=1e-9)
    assert recovered.arg_periapsis_rad == pytest.approx(NONDEGENERATE_ELEMENTS.arg_periapsis_rad, rel=1e-9)
    assert recovered.true_anomaly_rad == pytest.approx(NONDEGENERATE_ELEMENTS.true_anomaly_rad, rel=1e-9)


def test_state_to_elements_to_state_roundtrip():
    """The reverse round trip: a raw state vector survives the conversion out and back."""
    state = elements_to_state(NONDEGENERATE_ELEMENTS, MU_EARTH)
    elements = state_to_elements(state, MU_EARTH)
    recovered_state = elements_to_state(elements, MU_EARTH)

    np.testing.assert_allclose(recovered_state, state, rtol=1e-9)


def test_semi_major_axis_matches_specific_energy_from_m1():
    """
    Direct cross-check against M1: a = -mu/(2*epsilon) using the SAME
    specific_orbital_energy function already validated in M1 -- this
    is not a new, independent calculation, it's the same physics
    viewed through a different function, and both must agree.
    """
    state = elements_to_state(NONDEGENERATE_ELEMENTS, MU_EARTH)
    epsilon = specific_orbital_energy(state, MU_EARTH)
    expected_a = -MU_EARTH / (2.0 * epsilon)

    elements = state_to_elements(state, MU_EARTH)
    assert elements.semi_major_axis_km == pytest.approx(expected_a, rel=1e-12)


def test_circular_equatorial_orbit_does_not_crash():
    """
    The M1 nominal scenario is circular AND equatorial -- the double-
    singularity case. This must be handled gracefully (documented
    fallback values, correct flags), not raise or return NaN.
    """
    r0 = np.array([6778.137, 0.0, 0.0])
    v0 = np.array([0.0, 7.6685582, 0.0])
    state = np.concatenate([r0, v0])

    elements = state_to_elements(state, MU_EARTH)

    assert elements.is_circular is True
    assert elements.is_equatorial is True
    assert elements.eccentricity == pytest.approx(0.0, abs=1e-8)
    assert elements.inclination_rad == pytest.approx(0.0, abs=1e-8)
    assert not math.isnan(elements.arg_periapsis_rad)
    assert not math.isnan(elements.raan_rad)
    assert not math.isnan(elements.true_anomaly_rad)


def test_circular_equatorial_semi_major_axis_equals_radius():
    """For a circular orbit specifically, semi-major axis must equal the (constant) radius."""
    r0 = np.array([6778.137, 0.0, 0.0])
    v0 = np.array([0.0, 7.6685582, 0.0])
    state = np.concatenate([r0, v0])

    elements = state_to_elements(state, MU_EARTH)
    assert elements.semi_major_axis_km == pytest.approx(6778.137, rel=1e-6)


def test_inclined_circular_orbit_is_circular_but_not_equatorial():
    """
    A circular orbit that is NOT equatorial should trigger only the
    is_circular fallback, not the is_equatorial one -- confirms the two
    singularity checks are independent, not accidentally coupled.
    """
    circular_inclined = OrbitalElements(
        semi_major_axis_km=7000.0,
        eccentricity=0.0,
        inclination_rad=math.radians(51.6),
        raan_rad=math.radians(10.0),
        arg_periapsis_rad=0.0,
        true_anomaly_rad=math.radians(20.0),
        is_circular=True,
        is_equatorial=False,
    )
    state = elements_to_state(circular_inclined, MU_EARTH)
    elements = state_to_elements(state, MU_EARTH)

    assert elements.is_circular is True
    assert elements.is_equatorial is False
    assert elements.inclination_rad == pytest.approx(math.radians(51.6), rel=1e-6)
