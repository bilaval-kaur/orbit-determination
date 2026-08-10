"""
Integration test: cross-validates the hand-written RK4 propagator
against scipy.integrate.solve_ivp using the high-order adaptive DOP853
method, per docs/conventions.md Section 7.

This is an INTEGRATION test (not a physical validation test): it checks
that two independently-implemented propagation paths agree with each
other, not that either one obeys a specific physical law. If this test
fails, the bug is almost certainly in our hand-written RK4 or in
two_body_eom -- DOP853 is a mature, extensively validated implementation
and is trusted as the reference here.
"""

import numpy as np
from scipy.integrate import solve_ivp

from orbitdet.dynamics.integrators import propagate_rk4
from orbitdet.dynamics.two_body import two_body_eom

MU_EARTH = 398600.4418
R0 = np.array([6778.137, 0.0, 0.0])
V0 = np.array([0.0, 7.6685582, 0.0])
STATE0 = np.concatenate([R0, V0])
DURATION_S = 18000.0


def test_rk4_agrees_with_scipy_dop853_reference():
    """
    Propagate the same initial condition with both our RK4 and scipy's
    DOP853, using the SAME two_body_eom function for both (so we are
    validating the integrator, not accidentally comparing two different
    physics implementations). Final states must agree to within a tight
    tolerance relative to the orbit's characteristic scale.
    """
    _, rk4_states = propagate_rk4(
        two_body_eom, t0=0.0, state0=STATE0, step_s=10.0, duration_s=DURATION_S, mu=MU_EARTH
    )
    rk4_final = rk4_states[-1]

    reference = solve_ivp(
        two_body_eom,
        t_span=(0.0, DURATION_S),
        y0=STATE0,
        method="DOP853",
        args=(MU_EARTH,),
        rtol=1e-12,
        atol=1e-12,
    )
    reference_final = reference.y[:, -1]

    position_error_km = np.linalg.norm(rk4_final[0:3] - reference_final[0:3])
    velocity_error_km_s = np.linalg.norm(rk4_final[3:6] - reference_final[3:6])

    # Tolerance reflects RK4's O(h^4) truncation error at a 10s step over
    # ~3.2 orbits, compared against a near-exact (rtol/atol=1e-12) reference.
    assert position_error_km < 1.0, f"Position disagreement with DOP853: {position_error_km:.4f} km"
    assert velocity_error_km_s < 1e-3, f"Velocity disagreement with DOP853: {velocity_error_km_s:.6f} km/s"
