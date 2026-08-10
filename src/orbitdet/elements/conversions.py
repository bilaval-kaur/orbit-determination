"""
Conversions between the state vector [r, v] and classical (Keplerian)
orbital elements: semi-major axis, eccentricity, inclination, RAAN,
argument of periapsis, and true anomaly.

Both representations encode the same information; elements are the
more geometrically intuitive one, and are (mostly) constant for
unperturbed two-body motion -- only true anomaly changes with time.

Angles are stored INTERNALLY in radians (per docs/conventions.md); any
degree conversion happens only at a human-readable I/O boundary.

Singularities (see module-level discussion in docs/mathematics.md):
    - Circular orbits (e ~ 0): argument of periapsis and true anomaly
      are geometrically undefined (no unique periapsis to reference).
    - Equatorial orbits (i ~ 0 or 180 deg): RAAN is geometrically
      undefined (no unique ascending node).
This module detects both cases (via a tolerance on the relevant vector
magnitude) and falls back to well-documented conventions (angle -> 0.0)
rather than dividing by a near-zero magnitude and returning garbage or
NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Eccentricity is already dimensionless (0 for circular, approaching 1 for
# highly elliptical) -- a direct tolerance is dimensionally valid. Set
# looser than machine epsilon because REAL scenario inputs (e.g. a
# velocity truncated to 7 decimal places in a YAML config) introduce
# genuine, if tiny, non-exact eccentricity for an intended-circular
# orbit -- see docs/mathematics.md for a worked example of this.
_ECCENTRICITY_TOL = 1e-6

# The node vector N has units of km^2/s -- the SAME units as angular
# momentum h. Comparing N directly to a fixed absolute threshold is
# dimensionally wrong: a LEO orbit has h ~ 5e4 km^2/s while a GEO orbit
# has h ~ 1.3e5 km^2/s, so a single absolute cutoff can't correctly
# detect "near-equatorial" across different orbit scales. Instead we
# compare the DIMENSIONLESS ratio N/h to a tolerance, which is scale-
# invariant by construction.
_EQUATORIAL_RATIO_TOL = 1e-8


@dataclass
class OrbitalElements:
    """
    Classical (Keplerian) orbital elements. Angles in radians.

    is_circular / is_equatorial flag when a value that depends on a
    singular geometry (argument of periapsis / RAAN respectively) was
    set to a documented default (0.0) rather than being a genuinely
    computed, unambiguous value -- callers that care about the
    distinction should check these flags rather than assume every
    field is always meaningful.
    """

    semi_major_axis_km: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_periapsis_rad: float
    true_anomaly_rad: float
    is_circular: bool
    is_equatorial: bool


def state_to_elements(state: np.ndarray, mu: float) -> OrbitalElements:
    """
    Convert a state vector [rx,ry,rz,vx,vy,vz] to classical orbital elements.

    See docs/mathematics.md for the full step-by-step derivation this
    implementation follows.
    """
    r_vec = state[0:3]
    v_vec = state[3:6]

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    v_radial = np.dot(r_vec, v_vec) / r

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    inclination = math.acos(np.clip(h_vec[2] / h, -1.0, 1.0))

    z_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(z_hat, h_vec)
    n = np.linalg.norm(n_vec)

    is_equatorial = (n / h) < _EQUATORIAL_RATIO_TOL

    if is_equatorial:
        raan = 0.0  # Convention: undefined when orbital plane == equatorial plane.
    else:
        raan = math.acos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0:
            raan = 2 * math.pi - raan

    e_vec = (1.0 / mu) * ((v**2 - mu / r) * r_vec - v_radial * r * v_vec)
    e = np.linalg.norm(e_vec)

    is_circular = e < _ECCENTRICITY_TOL

    if is_circular or is_equatorial:
        arg_periapsis = 0.0  # Convention: undefined without a unique node and/or periapsis.
    else:
        arg_periapsis = math.acos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            arg_periapsis = 2 * math.pi - arg_periapsis

    if is_circular:
        # Fall back to angle from the ascending node (or from x-axis, if
        # also equatorial) directly to the spacecraft -- still a
        # well-defined "where is it right now" even without a periapsis.
        if is_equatorial:
            reference_vec = np.array([1.0, 0.0, 0.0])
        else:
            reference_vec = n_vec
        ref_norm = np.linalg.norm(reference_vec)
        true_anomaly = math.acos(np.clip(np.dot(reference_vec, r_vec) / (ref_norm * r), -1.0, 1.0))
        if r_vec[1] < 0:
            true_anomaly = 2 * math.pi - true_anomaly
    else:
        true_anomaly = math.acos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if v_radial < 0:
            true_anomaly = 2 * math.pi - true_anomaly

    specific_energy = v**2 / 2.0 - mu / r
    semi_major_axis = -mu / (2.0 * specific_energy)

    return OrbitalElements(
        semi_major_axis_km=float(semi_major_axis),
        eccentricity=float(e),
        inclination_rad=float(inclination),
        raan_rad=float(raan),
        arg_periapsis_rad=float(arg_periapsis),
        true_anomaly_rad=float(true_anomaly),
        is_circular=bool(is_circular),
        is_equatorial=bool(is_equatorial),
    )


def elements_to_state(elements: OrbitalElements, mu: float) -> np.ndarray:
    """
    Convert classical orbital elements back to a state vector.

    Builds position/velocity in the PERIFOCAL frame (P toward periapsis,
    Q 90 degrees ahead in the direction of motion, W along the angular
    momentum vector -- the orbit's own natural, non-rotating coordinate
    system) using the orbit equation, then rotates into ECI via the
    classical 3-1-3 Euler sequence: rotate by RAAN about z, then by
    inclination about the (new) x-axis (the line of nodes), then by
    argument of periapsis about the (new) z-axis (the angular momentum
    direction). See docs/mathematics.md for the full derivation.
    """
    a = elements.semi_major_axis_km
    e = elements.eccentricity
    i = elements.inclination_rad
    raan = elements.raan_rad
    argp = elements.arg_periapsis_rad
    nu = elements.true_anomaly_rad

    p = a * (1.0 - e**2)  # semi-latus rectum
    r_mag = p / (1.0 + e * math.cos(nu))
    h = math.sqrt(mu * p)

    r_pf = r_mag * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = (mu / h) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    def rot_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rot_x(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    # 3-1-3 sequence, composed right-to-left: first rotate by argp about
    # z (perifocal -> "ascending-node-aligned" frame), then by i about
    # x (tilts into the orbital plane), then by raan about z (rotates
    # the node to its true position in the reference frame).
    rotation = rot_z(raan) @ rot_x(i) @ rot_z(argp)

    r_eci = rotation @ r_pf
    v_eci = rotation @ v_pf

    return np.concatenate([r_eci, v_eci])
