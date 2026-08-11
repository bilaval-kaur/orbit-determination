"""
Experiment framework: wires M1-M8 into one reusable, config-driven
pipeline, producing structured metrics for controlled experiments.

Every experiment is fully specified by an ExperimentConfig (per
docs/conventions.md Section 6 -- explicit seed, explicit noise levels,
no hidden state) and produces an ExperimentResult with comparable
metrics across experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitdet.detection.anomaly import detect_with_persistence, nis_threshold
from orbitdet.dynamics.integrators import propagate_rk4
from orbitdet.dynamics.two_body import two_body_eom
from orbitdet.estimation.consistency import compute_nees, compute_nis, consistency_interval
from orbitdet.estimation.ekf import ekf_predict, ekf_update
from orbitdet.measurements.models import generate_measurement_series, position_measurement_matrix
from orbitdet.optimization.planner import solve_minimum_fuel_correction

MU_EARTH = 398600.4418
NOMINAL_STATE0 = np.array([6778.137, 0.0, 0.0, 0.0, 7.6685582, 0.0])


@dataclass
class ExperimentConfig:
    """Fully specifies one reproducible experiment."""

    name: str
    description: str
    seed: int
    duration_s: float = 3000.0
    dt: float = 10.0

    measurement_sigma_km: float = 0.5
    process_noise_q_pos: float = 3e-8
    process_noise_q_vel: float = 3e-11

    initial_position_error_km: float = 0.0
    initial_velocity_error_km_s: float = 0.0

    inject_perturbation: bool = False
    perturbation_dv_km_s: float = 0.1
    perturbation_time_s: float = 1500.0

    run_correction: bool = False
    correction_duration_s: float = 1000.0
    correction_tolerance_km: float = 0.5
    # Steps to wait AFTER confirmed detection before planning the
    # correction. Discovered necessary during development: with
    # position-only measurements, the EKF's VELOCITY estimate converges
    # much more slowly than its position estimate after a sudden
    # perturbation (velocity is only indirectly observable -- an
    # observability limitation, not a bug).
    #
    # IMPORTANT, HONEST LIMITATION: this settling delay is only a
    # PARTIAL mitigation, not a full fix. Empirically, even after this
    # many steps, the velocity estimate remains substantially biased
    # (in testing, ~150s of settling reduced a 0.1 km/s velocity error
    # to only ~0.097 km/s -- barely improved). This is because the
    # small process noise Q, carefully tuned in M5 for NOMINAL filter
    # consistency, makes the filter deliberately slow to revise its
    # velocity belief -- exactly the wrong property when a genuine,
    # unmodeled change has actually occurred. This is a real, known
    # tension in estimation theory: tuning for nominal-condition
    # consistency and tuning for fast post-anomaly adaptation pull in
    # opposite directions. A proper fix (adaptive process noise /
    # covariance inflation triggered on detection, letting the filter
    # temporarily "listen" more aggressively to new measurements) is
    # flagged as a Future Improvement rather than implemented here, to
    # avoid scope creep -- but the resulting maneuver_delta_v_km_s in
    # perturbation experiments should be understood as likely an
    # UNDERESTIMATE of the true required correction, not a fully-
    # converged optimal answer.
    correction_settling_steps: int = 15


@dataclass
class ExperimentResult:
    """Structured, comparable metrics from running one ExperimentConfig."""

    config_name: str
    mean_position_error_km: float
    nees_mean: float
    nees_consistent: bool
    nis_mean: float
    nis_consistent: bool
    anomaly_detected: bool
    detection_latency_steps: int | None
    maneuver_delta_v_km_s: float | None
    post_maneuver_error_km: float | None


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Execute one full experiment: build the true trajectory (optionally
    with an injected perturbation), generate noisy measurements, run
    the EKF, evaluate consistency (M5), run anomaly detection (M6), and
    optionally solve and apply a corrective maneuver (M8).
    """
    rng = np.random.default_rng(config.seed)

    if config.inject_perturbation:
        _, true_before = propagate_rk4(
            two_body_eom, 0.0, NOMINAL_STATE0, config.dt, config.perturbation_time_s, MU_EARTH
        )
        perturbed_state = true_before[-1].copy()
        perturbed_state[3:6] += np.array([config.perturbation_dv_km_s, 0.0, 0.0])
        remaining_s = config.duration_s - config.perturbation_time_s
        _, true_after = propagate_rk4(
            two_body_eom, 0.0, perturbed_state, config.dt, remaining_s, MU_EARTH
        )
        true_states = np.vstack([true_before, true_after[1:]])
    else:
        _, true_states = propagate_rk4(
            two_body_eom, 0.0, NOMINAL_STATE0, config.dt, config.duration_s, MU_EARTH
        )

    measurements = generate_measurement_series(true_states, config.measurement_sigma_km, rng)

    H = position_measurement_matrix()
    R = np.eye(3) * config.measurement_sigma_km**2
    Q = np.diag(
        [config.process_noise_q_pos] * 3 + [config.process_noise_q_vel] * 3
    )

    initial_offset = np.zeros(6)
    initial_offset[0:3] = config.initial_position_error_km / np.sqrt(3)
    initial_offset[3:6] = config.initial_velocity_error_km_s / np.sqrt(3)

    state = NOMINAL_STATE0 + initial_offset
    P = np.eye(6) * 1.0

    position_errors = []
    nees_values = []
    nis_values = []
    state_history = []  # snapshot of the filter's estimate at every step,
    # needed so a corrective maneuver can be planned from the estimate
    # that existed AT DETECTION TIME, not just the final loop state.

    for k in range(1, len(true_states)):
        state, P = ekf_predict(state, P, config.dt, MU_EARTH, Q)
        state, P, innovation, S = ekf_update(state, P, measurements[k], H, R)

        position_errors.append(float(np.linalg.norm(state[0:3] - true_states[k, 0:3])))
        nees_values.append(compute_nees(true_states[k], state, P))
        nis_values.append(compute_nis(innovation, S))
        state_history.append(state.copy())

    position_errors = np.array(position_errors)
    nees_values = np.array(nees_values)
    nis_values = np.array(nis_values)

    # Discard initial transient before assessing steady-state consistency.
    transient = 10
    nees_steady = nees_values[transient:]
    nis_steady = nis_values[transient:]

    nees_lower, nees_upper = consistency_interval(dof=6, n_samples=len(nees_steady))
    nis_lower, nis_upper = consistency_interval(dof=3, n_samples=len(nis_steady))
    nees_mean = float(np.mean(nees_steady))
    nis_mean = float(np.mean(nis_steady))

    threshold = nis_threshold(dof=3, alpha=0.01)
    confirmed = detect_with_persistence(nis_values, threshold)
    anomaly_detected = bool(np.any(confirmed))

    detection_latency = None
    if config.inject_perturbation and anomaly_detected:
        injection_index = int(config.perturbation_time_s / config.dt) - 1
        post_injection_confirmed = confirmed[injection_index:]
        if np.any(post_injection_confirmed):
            detection_latency = int(np.argmax(post_injection_confirmed))

    maneuver_dv = None
    post_maneuver_error = None
    if config.run_correction and anomaly_detected and detection_latency is not None:
        detect_index = injection_index + detection_latency
        # Wait for the filter's velocity estimate to settle before
        # planning the correction (see correction_settling_steps
        # docstring above) -- clamped to not run past the end of the
        # available trajectory.
        plan_index = min(detect_index + config.correction_settling_steps, len(state_history) - 1)

        # Reference: what the mission plan says (undisturbed propagation).
        target_time_s = (plan_index + 1) * config.dt + config.correction_duration_s
        _, reference_states = propagate_rk4(
            two_body_eom, 0.0, NOMINAL_STATE0, config.dt, target_time_s, MU_EARTH
        )
        target_position = reference_states[-1, 0:3]

        # Use the FILTER's estimate at the settled planning time (from
        # the snapshotted history) as the current state to correct
        # from -- matching real usage, where true state is never
        # available and the estimate used must be one that has had
        # time to reflect the actual perturbation.
        current_estimate = state_history[plan_index]

        solution = solve_minimum_fuel_correction(
            current_estimate, target_position,
            correction_duration_s=config.correction_duration_s,
            mu=MU_EARTH, tolerance_km=config.correction_tolerance_km,
        )
        maneuver_dv = solution.delta_v_cost_km_s
        post_maneuver_error = solution.achieved_position_error_km

    return ExperimentResult(
        config_name=config.name,
        mean_position_error_km=float(np.mean(position_errors)),
        nees_mean=nees_mean,
        nees_consistent=nees_lower < nees_mean < nees_upper,
        nis_mean=nis_mean,
        nis_consistent=nis_lower < nis_mean < nis_upper,
        anomaly_detected=anomaly_detected,
        detection_latency_steps=detection_latency,
        maneuver_delta_v_km_s=maneuver_dv,
        post_maneuver_error_km=post_maneuver_error,
    )
