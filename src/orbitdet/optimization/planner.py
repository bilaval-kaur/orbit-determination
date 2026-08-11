"""
Minimum-fuel maneuver planning via constrained optimization.

Objective formulation (see docs/mathematics.md for full justification,
and the M0 planning discussion for why this replaces a weighted-sum
objective): minimize Delta-v magnitude subject to a hard constraint on
post-maneuver position error, rather than combining error/cost/time
into one arbitrary-weighted scalar with mismatched units.

    minimize    ||delta_v_rtn||
    subject to  position_error(delta_v_rtn) <= tolerance

Internally minimizes ||delta_v_rtn||^2 (smooth, same minimizer as the
true norm since sqrt is monotonic for x>=0), reporting the actual
Delta-v magnitude as the final cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from orbitdet.maneuver.impulsive import apply_maneuver_and_propagate


@dataclass
class ManeuverSolution:
    """Result of solving the minimum-fuel correction problem."""

    delta_v_rtn: np.ndarray
    delta_v_cost_km_s: float
    achieved_position_error_km: float
    constraint_satisfied: bool
    optimizer_success: bool
    n_iterations: int


def correction_position_error(
    delta_v_rtn: np.ndarray,
    current_state: np.ndarray,
    target_position: np.ndarray,
    correction_duration_s: float,
    step_s: float,
    mu: float,
) -> float:
    """
    The constraint function: apply a candidate burn, propagate forward
    by correction_duration_s, and return the resulting position error
    relative to the target (reference trajectory's) position at that
    future time.
    """
    _, states = apply_maneuver_and_propagate(
        current_state, delta_v_rtn, correction_duration_s, step_s, mu
    )
    final_position = states[-1, 0:3]
    return float(np.linalg.norm(final_position - target_position))


def solve_minimum_fuel_correction(
    current_state: np.ndarray,
    target_position: np.ndarray,
    correction_duration_s: float,
    mu: float,
    tolerance_km: float,
    step_s: float = 10.0,
) -> ManeuverSolution:
    """
    Solve for the minimum-Delta-v single impulsive burn (in RTN
    components) that brings the spacecraft's position, after
    correction_duration_s of propagation, within tolerance_km of
    target_position.

    Uses scipy.optimize.minimize with SLSQP (Sequential Least Squares
    Programming) -- a standard constrained nonlinear optimization
    method -- with one inequality constraint (position error <= tolerance).
    """
    # Heuristic initial guess: a rough order-of-magnitude estimate,
    # dv ~ position_error / correction_time, applied in the TANGENTIAL
    # direction -- the efficient burn direction per M7's finding that
    # tangential burns dominate energy change at first order. This is a
    # legitimate back-of-envelope estimate used in real trajectory
    # correction planning, not an arbitrary starting point.
    baseline_error = correction_position_error(
        np.zeros(3), current_state, target_position, correction_duration_s, step_s, mu
    )
    initial_guess = np.array([0.0, baseline_error / correction_duration_s, 0.0])

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))  # ||delta_v||^2 -- smooth, same minimizer as ||delta_v||

    def constraint(x: np.ndarray) -> float:
        error = correction_position_error(
            x, current_state, target_position, correction_duration_s, step_s, mu
        )
        return tolerance_km - error  # scipy 'ineq' convention: must be >= 0

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": constraint}],
        options={"maxiter": 200, "ftol": 1e-8},
    )

    delta_v_rtn = result.x
    achieved_error = correction_position_error(
        delta_v_rtn, current_state, target_position, correction_duration_s, step_s, mu
    )

    return ManeuverSolution(
        delta_v_rtn=delta_v_rtn,
        delta_v_cost_km_s=float(np.linalg.norm(delta_v_rtn)),
        achieved_position_error_km=achieved_error,
        constraint_satisfied=achieved_error <= tolerance_km * 1.01,  # small numerical margin
        optimizer_success=bool(result.success),
        n_iterations=int(result.nit),
    )


def grid_search_correction(
    current_state: np.ndarray,
    target_position: np.ndarray,
    correction_duration_s: float,
    mu: float,
    tolerance_km: float,
    step_s: float = 10.0,
    search_range_km_s: float = 1.0,
    resolution: int = 9,
) -> ManeuverSolution:
    """
    Coarse grid search over the FULL 3D RTN Delta-v space, as an
    independent baseline to sanity-check the optimizer -- confirms
    SLSQP found a genuinely good solution rather than a poor local
    optimum. An earlier version searched only the tangential direction;
    that was corrected after discovering it could not reach arbitrary
    3D position targets (a single-axis search only spans a 1-parameter
    family of achievable position errors, insufficient for a generic
    3D target displacement). Cost scales as resolution^3, so resolution
    is kept modest -- this is a coarse cross-check, not a general-
    purpose optimizer.
    """
    candidates = np.linspace(-search_range_km_s, search_range_km_s, resolution)

    best_cost = np.inf
    best_dv = np.zeros(3)
    best_error = np.inf

    for dv_r in candidates:
        for dv_t in candidates:
            for dv_n in candidates:
                dv_rtn = np.array([dv_r, dv_t, dv_n])
                cost = float(np.linalg.norm(dv_rtn))
                if cost >= best_cost:
                    continue  # cheap pre-filter before the expensive propagation call
                error = correction_position_error(
                    dv_rtn, current_state, target_position, correction_duration_s, step_s, mu
                )
                if error <= tolerance_km:
                    best_cost = cost
                    best_dv = dv_rtn
                    best_error = error

    return ManeuverSolution(
        delta_v_rtn=best_dv,
        delta_v_cost_km_s=best_cost if best_cost < np.inf else float("nan"),
        achieved_position_error_km=best_error,
        constraint_satisfied=best_cost < np.inf,
        optimizer_success=best_cost < np.inf,
        n_iterations=resolution**3,
    )
