"""
Runs the 7 controlled experiments defined in the project's original
scope, using the M9 experiment framework (src/orbitdet/scenarios/
experiment_runner.py), and produces a structured summary table.

Each experiment isolates ONE variable relative to a shared nominal
baseline, so differences in the results are attributable to that one
change -- the actual point of a "controlled" experiment.
"""

import pandas as pd

from orbitdet.scenarios.experiment_runner import ExperimentConfig, run_experiment

EXPERIMENTS = [
    ExperimentConfig(
        name="1_low_measurement_noise",
        description="Baseline with tight measurement precision (sigma=0.1 km).",
        seed=1,
        measurement_sigma_km=0.1,
    ),
    ExperimentConfig(
        name="2_high_measurement_noise",
        description="Same baseline, but with degraded sensor precision (sigma=2.0 km).",
        seed=1,
        measurement_sigma_km=2.0,
    ),
    ExperimentConfig(
        name="3_low_process_noise",
        description="Process noise reduced 100x below the M5-validated tuning -- expected to "
        "reveal filter overconfidence (NEES/NIS inflated), echoing M5's own discovery.",
        seed=1,
        process_noise_q_pos=3e-10,
        process_noise_q_vel=3e-13,
    ),
    ExperimentConfig(
        name="4_large_initial_state_error",
        description="Filter initialized 5 km / 0.05 km/s away from truth -- tests convergence "
        "from a poor starting guess.",
        seed=1,
        initial_position_error_km=5.0,
        initial_velocity_error_km_s=0.05,
    ),
    ExperimentConfig(
        name="5_unexpected_perturbation",
        description="An unplanned 0.1 km/s velocity kick at t=1500s, with detection but no "
        "correction -- isolates M6's detection behavior.",
        seed=42,
        inject_perturbation=True,
        perturbation_dv_km_s=0.1,
        perturbation_time_s=1500.0,
    ),
    ExperimentConfig(
        name="6_perturbation_with_correction",
        description="Same perturbation as Experiment 5, but with a corrective maneuver planned "
        "and evaluated (M8) -- NOTE: see Limitations regarding post-detection velocity "
        "estimate convergence lag; reported Delta-v likely underestimates the true "
        "required correction.",
        seed=42,
        inject_perturbation=True,
        perturbation_dv_km_s=0.1,
        perturbation_time_s=1500.0,
        run_correction=True,
        correction_duration_s=1000.0,
        correction_tolerance_km=0.5,
    ),
    ExperimentConfig(
        name="7_tighter_correction_tolerance",
        description="Same as Experiment 6, but with a tighter correction tolerance (0.05 km "
        "vs 0.5 km) -- expected to require a larger corrective Delta-v, per the "
        "constrained-optimization property validated in M8.",
        seed=42,
        inject_perturbation=True,
        perturbation_dv_km_s=0.1,
        perturbation_time_s=1500.0,
        run_correction=True,
        correction_duration_s=1000.0,
        correction_tolerance_km=0.05,
    ),
]


def main() -> None:
    results = [run_experiment(config) for config in EXPERIMENTS]

    rows = []
    for config, result in zip(EXPERIMENTS, results):
        rows.append(
            {
                "experiment": result.config_name,
                "mean_pos_error_km": round(result.mean_position_error_km, 4),
                "nees_mean": round(result.nees_mean, 2),
                "nees_consistent": result.nees_consistent,
                "nis_mean": round(result.nis_mean, 2),
                "nis_consistent": result.nis_consistent,
                "anomaly_detected": result.anomaly_detected,
                "detection_latency_steps": result.detection_latency_steps,
                "maneuver_dv_km_s": (
                    round(result.maneuver_delta_v_km_s, 5) if result.maneuver_delta_v_km_s is not None else None
                ),
                "post_maneuver_error_km": (
                    round(result.post_maneuver_error_km, 4) if result.post_maneuver_error_km is not None else None
                ),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))

    df.to_csv("experiments/results_summary.csv", index=False)
    print("\nSaved to experiments/results_summary.csv")


if __name__ == "__main__":
    main()
