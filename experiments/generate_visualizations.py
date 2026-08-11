"""
Generates the project's core visualization figures, saved as PNG files
to docs/figures/. Each figure is designed to answer a specific
engineering question, per the project's original visualization
requirements -- not decorative plots.

Run with: python3 experiments/generate_visualizations.py
"""

import matplotlib.pyplot as plt
import numpy as np

from orbitdet.detection.anomaly import detect_with_persistence, nis_threshold
from orbitdet.dynamics.integrators import propagate_rk4
from orbitdet.dynamics.two_body import two_body_eom
from orbitdet.estimation.consistency import compute_nis
from orbitdet.estimation.ekf import ekf_predict, ekf_update
from orbitdet.measurements.models import generate_measurement_series, position_measurement_matrix
from orbitdet.optimization.planner import solve_minimum_fuel_correction

MU_EARTH = 398600.4418
STATE0 = np.array([6778.137, 0.0, 0.0, 0.0, 7.6685582, 0.0])

# Mission Control's dashboard color palette, reused deliberately for
# visual continuity across both portfolio projects.
COLOR_TRUE = "#22d3ee"      # cyan -- real/truth
COLOR_ESTIMATE = "#eab308"  # amber -- estimated/simulated
COLOR_REFERENCE = "#8b949e"  # gray -- reference/nominal
COLOR_ALERT = "#ef4444"     # red -- anomaly/critical
BG_COLOR = "#0d1117"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#1f2a36"


def _style_axis(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def figure_1_orbit_determination_performance():
    """
    2x2: true vs. estimated ground track, position error with 3-sigma
    bound, velocity error with 3-sigma bound, NIS time series with
    chi-squared threshold. Answers: "does the filter track the true
    orbit, and does it honestly know how well it's doing?"
    """
    rng = np.random.default_rng(7)
    sigma_km = 0.5
    dt = 10.0
    duration_s = 3000.0

    _, true_states = propagate_rk4(two_body_eom, 0.0, STATE0, dt, duration_s, MU_EARTH)
    measurements = generate_measurement_series(true_states, sigma_km, rng)

    H = position_measurement_matrix()
    R = np.eye(3) * sigma_km**2
    Q = np.diag([3e-8, 3e-8, 3e-8, 3e-11, 3e-11, 3e-11])

    state = STATE0 + np.array([2.0, -2.0, 1.0, 0.001, -0.001, 0.0])
    P = np.eye(6) * 4.0

    est_states = [state.copy()]
    pos_errors, vel_errors = [], []
    pos_3sigma, vel_3sigma = [], []
    nis_series = []

    for k in range(1, len(true_states)):
        state, P = ekf_predict(state, P, dt, MU_EARTH, Q)
        state, P, innovation, S = ekf_update(state, P, measurements[k], H, R)
        est_states.append(state.copy())
        pos_errors.append(np.linalg.norm(state[0:3] - true_states[k, 0:3]))
        vel_errors.append(np.linalg.norm(state[3:6] - true_states[k, 3:6]))
        pos_3sigma.append(3 * np.sqrt(np.trace(P[0:3, 0:3])))
        vel_3sigma.append(3 * np.sqrt(np.trace(P[3:6, 3:6])))
        nis_series.append(compute_nis(innovation, S))

    est_states = np.array(est_states)
    times_min = np.arange(len(pos_errors)) * dt / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=BG_COLOR)

    ax = axes[0, 0]
    ax.plot(true_states[:, 0], true_states[:, 1], color=COLOR_TRUE, linewidth=2, label="True")
    ax.plot(est_states[:, 0], est_states[:, 1], color=COLOR_ESTIMATE, linewidth=1, linestyle="--", label="Estimated")
    ax.set_xlabel("ECI X (km)")
    ax.set_ylabel("ECI Y (km)")
    ax.set_title("True vs. Estimated Trajectory")
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_aspect("equal")
    _style_axis(ax)

    ax = axes[0, 1]
    ax.plot(times_min, pos_errors, color=COLOR_ESTIMATE, linewidth=1.2, label="Actual error")
    ax.plot(times_min, pos_3sigma, color=COLOR_TRUE, linewidth=1, linestyle=":", label="Filter 3-sigma bound")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Position error (km)")
    ax.set_title("Position Estimation Error")
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    _style_axis(ax)

    ax = axes[1, 0]
    ax.plot(times_min, vel_errors, color=COLOR_ESTIMATE, linewidth=1.2, label="Actual error")
    ax.plot(times_min, vel_3sigma, color=COLOR_TRUE, linewidth=1, linestyle=":", label="Filter 3-sigma bound")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Velocity error (km/s)")
    ax.set_title("Velocity Estimation Error")
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    _style_axis(ax)

    ax = axes[1, 1]
    threshold = nis_threshold(dof=3, alpha=0.01)
    ax.plot(times_min, nis_series, color=COLOR_ESTIMATE, linewidth=1, label="NIS")
    ax.axhline(threshold, color=COLOR_ALERT, linewidth=1, linestyle="--", label=f"Anomaly threshold ({threshold:.1f})")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("NIS")
    ax.set_title("Innovation Consistency (NIS)")
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    _style_axis(ax)

    fig.suptitle("Orbit Determination Performance (Nominal Scenario)", color=TEXT_COLOR, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig("docs/figures/01_orbit_determination_performance.png", dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print("Saved docs/figures/01_orbit_determination_performance.png")


def figure_2_anomaly_detection_and_correction():
    """
    2x1: NIS spike at an injected perturbation with confirmed detection
    marked, and the pre/post-maneuver trajectory showing the correction
    converging back toward the reference. Answers: "does the system
    catch a real problem, and can it fix it?"
    """
    rng = np.random.default_rng(42)
    sigma_km = 0.5
    dt = 10.0
    injection_step = 150
    duration_s = 3000.0

    _, true_before = propagate_rk4(two_body_eom, 0.0, STATE0, dt, injection_step * dt, MU_EARTH)
    perturbed_state = true_before[-1].copy()
    perturbed_state[3:6] += np.array([0.1, 0.0, 0.0])
    _, true_after = propagate_rk4(
        two_body_eom, 0.0, perturbed_state, dt, duration_s - injection_step * dt, MU_EARTH
    )
    true_states = np.vstack([true_before, true_after[1:]])
    measurements = generate_measurement_series(true_states, sigma_km, rng)

    H = position_measurement_matrix()
    R = np.eye(3) * sigma_km**2
    Q = np.diag([3e-8, 3e-8, 3e-8, 3e-11, 3e-11, 3e-11])
    state, P = STATE0.copy(), np.eye(6) * 1.0

    nis_series = []
    for k in range(1, len(true_states)):
        state, P = ekf_predict(state, P, dt, MU_EARTH, Q)
        state, P, innovation, S = ekf_update(state, P, measurements[k], H, R)
        nis_series.append(compute_nis(innovation, S))
    nis_series = np.array(nis_series)

    threshold = nis_threshold(dof=3, alpha=0.01)
    confirmed = detect_with_persistence(nis_series, threshold)
    times_min = np.arange(len(nis_series)) * dt / 60.0
    injection_min = injection_step * dt / 60.0
    detect_idx = np.argmax(confirmed[injection_step - 1 :]) + injection_step - 1

    # Reference (undisturbed) trajectory, evaluated at EXACTLY the same
    # future time the correction will arrive at -- injection_step*dt +
    # correction_duration_s. An earlier version used a mismatched,
    # independently-chosen reference duration here, causing the
    # optimizer to aim for a target 1000 seconds away from where the
    # corrected trajectory would actually be, producing a physically
    # absurd multi-km/s "correction." Target time and propagation
    # window must always agree exactly.
    correction_duration_s = 1000.0
    target_time_s = injection_step * dt + correction_duration_s
    _, reference_states = propagate_rk4(two_body_eom, 0.0, STATE0, dt, target_time_s, MU_EARTH)
    target_position = reference_states[-1, 0:3]

    solution = solve_minimum_fuel_correction(
        perturbed_state, target_position, correction_duration_s, MU_EARTH, tolerance_km=0.5
    )
    _, corrected_states = apply_maneuver_and_propagate_states(
        perturbed_state, solution.delta_v_rtn, correction_duration_s, dt, MU_EARTH
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_COLOR)

    ax = axes[0]
    ax.plot(times_min, nis_series, color=COLOR_ESTIMATE, linewidth=1, label="NIS")
    ax.axhline(threshold, color=COLOR_ALERT, linewidth=1, linestyle="--", label="Anomaly threshold")
    ax.axvline(injection_min, color=COLOR_REFERENCE, linewidth=1, linestyle=":", label="Perturbation injected")
    ax.scatter([times_min[detect_idx]], [nis_series[detect_idx]], color=COLOR_ALERT, zorder=5, label="Confirmed detection")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("NIS")
    ax.set_title("Anomaly Detection")
    ax.set_xlim(injection_min - 5, injection_min + 10)
    ax.set_ylim(0, min(nis_series[injection_step - 5 : injection_step + 20].max() * 1.1, 500))
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8, loc="upper left")
    _style_axis(ax)

    ax = axes[1]
    ax.plot(reference_states[:, 0], reference_states[:, 1], color=COLOR_TRUE, linewidth=2, label="Reference (undisturbed)")
    ax.plot(true_after[:, 0], true_after[:, 1], color=COLOR_ALERT, linewidth=1.2, linestyle="--", label="Deviated (uncorrected)")
    ax.plot(corrected_states[:, 0], corrected_states[:, 1], color=COLOR_ESTIMATE, linewidth=1.5, label="Corrected trajectory")
    ax.scatter([perturbed_state[0]], [perturbed_state[1]], color=COLOR_REFERENCE, zorder=5, s=40, label="Maneuver point")
    ax.set_xlabel("ECI X (km)")
    ax.set_ylabel("ECI Y (km)")
    ax.set_title(f"Correction Maneuver (\u0394v = {solution.delta_v_cost_km_s*1000:.1f} m/s)")
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    _style_axis(ax)

    fig.suptitle("Anomaly Detection & Corrective Maneuver", color=TEXT_COLOR, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("docs/figures/02_anomaly_detection_and_correction.png", dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print("Saved docs/figures/02_anomaly_detection_and_correction.png")


def figure_3_experiment_comparison():
    """
    Bar charts from M9's experiment results: position error across
    noise/process-noise variations, and Delta-v cost across correction
    tolerance variations. Answers: "do the seven controlled experiments
    behave as the underlying theory predicts?"
    """
    import pandas as pd

    df = pd.read_csv("experiments/results_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_COLOR)

    ax = axes[0]
    subset = df[df["experiment"].isin([
        "1_low_measurement_noise", "2_high_measurement_noise",
        "3_low_process_noise", "4_large_initial_state_error",
    ])]
    labels = [n.split("_", 1)[1].replace("_", "\n") for n in subset["experiment"]]
    colors = [COLOR_ALERT if not c else COLOR_ESTIMATE for c in subset["nees_consistent"]]
    ax.bar(labels, subset["mean_pos_error_km"], color=colors)
    ax.set_ylabel("Mean position error (km)")
    ax.set_title("Experiments 1-4: Position Error\n(red = NEES inconsistent)")
    _style_axis(ax)

    ax = axes[1]
    subset2 = df[df["experiment"].isin(["6_perturbation_with_correction", "7_tighter_correction_tolerance"])]
    labels2 = ["Loose tolerance\n(0.5 km)", "Tight tolerance\n(0.05 km)"]
    ax.bar(labels2, subset2["maneuver_dv_km_s"] * 1000, color=COLOR_TRUE)
    ax.set_ylabel("Correction Delta-v (m/s)")
    ax.set_title("Experiments 6-7: Tolerance vs. Cost\n(tighter tolerance -> more Delta-v)")
    _style_axis(ax)

    fig.suptitle("Controlled Experiment Comparison", color=TEXT_COLOR, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("docs/figures/03_experiment_comparison.png", dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print("Saved docs/figures/03_experiment_comparison.png")


def apply_maneuver_and_propagate_states(state, delta_v_rtn, duration_s, dt, mu):
    """Local helper mirroring M7's apply_maneuver_and_propagate, kept here to avoid a circular import in this script."""
    from orbitdet.maneuver.impulsive import apply_maneuver_and_propagate
    return apply_maneuver_and_propagate(state, delta_v_rtn, duration_s, dt, mu)


if __name__ == "__main__":
    figure_1_orbit_determination_performance()
    figure_2_anomaly_detection_and_correction()
    figure_3_experiment_comparison()
