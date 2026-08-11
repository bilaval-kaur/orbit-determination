"""
Tests for the experiment framework.

Focused on framework mechanics and the most important CROSS-MILESTONE
behavioral checks -- not re-validating individual components (M1-M8
already do that exhaustively), but confirming they compose correctly
when wired together as configurable experiments.
"""

import numpy as np

from orbitdet.scenarios.experiment_runner import ExperimentConfig, run_experiment


def test_experiment_is_reproducible_with_same_seed():
    config = ExperimentConfig(name="repro_test", description="", seed=5)
    result1 = run_experiment(config)
    result2 = run_experiment(config)

    assert result1.mean_position_error_km == result2.mean_position_error_km
    assert result1.nees_mean == result2.nees_mean


def test_higher_measurement_noise_produces_larger_position_error():
    """Same variable-isolation check as Experiment 1 vs 2 -- confirms the framework's noise wiring is correct."""
    low_noise = ExperimentConfig(name="low", description="", seed=1, measurement_sigma_km=0.1)
    high_noise = ExperimentConfig(name="high", description="", seed=1, measurement_sigma_km=2.0)

    result_low = run_experiment(low_noise)
    result_high = run_experiment(high_noise)

    assert result_high.mean_position_error_km > result_low.mean_position_error_km


def test_low_process_noise_reproduces_m5_overconfidence_finding():
    """
    Independent confirmation, through this new framework, of M5's core
    discovery: process noise set far below the validated tuning makes
    the filter overconfident (NEES inflated well above the theoretical
    mean of 6). This is not a new finding -- it's the same physical
    phenomenon (linearization error unabsorbed by too-small Q) surfacing
    again through a completely different code path, which is itself a
    valuable internal consistency check on the whole project.
    """
    low_q_config = ExperimentConfig(
        name="low_q", description="", seed=1,
        process_noise_q_pos=3e-10, process_noise_q_vel=3e-13,  # 100x below M5's validated Q
    )
    result = run_experiment(low_q_config)

    assert result.nees_mean > 6.0
    assert not result.nees_consistent


def test_no_perturbation_scenario_does_not_trigger_anomaly_detection():
    """A genuinely nominal scenario (no injected perturbation) should not produce a confirmed anomaly."""
    config = ExperimentConfig(name="nominal", description="", seed=3, inject_perturbation=False)
    result = run_experiment(config)

    assert result.anomaly_detected is False
    assert result.detection_latency_steps is None


def test_perturbation_scenario_triggers_detection_with_reasonable_latency():
    """A real injected perturbation should be detected, with latency in the same ballpark as M6's finding (~3 steps)."""
    config = ExperimentConfig(
        name="perturbed", description="", seed=42,
        inject_perturbation=True, perturbation_dv_km_s=0.1, perturbation_time_s=1500.0,
    )
    result = run_experiment(config)

    assert result.anomaly_detected is True
    assert result.detection_latency_steps is not None
    assert result.detection_latency_steps < 10  # should detect within a handful of steps, not lag indefinitely


def test_tighter_correction_tolerance_requires_at_least_as_much_delta_v():
    """
    The M8 constrained-optimization property (tighter tolerance -> Delta-v
    can only increase or stay the same), now confirmed through the FULL
    integrated detect-then-correct pipeline, not just the isolated
    optimizer test from M8.
    """
    loose_config = ExperimentConfig(
        name="loose_tol", description="", seed=42,
        inject_perturbation=True, perturbation_dv_km_s=0.1, perturbation_time_s=1500.0,
        run_correction=True, correction_tolerance_km=0.5,
    )
    tight_config = ExperimentConfig(
        name="tight_tol", description="", seed=42,
        inject_perturbation=True, perturbation_dv_km_s=0.1, perturbation_time_s=1500.0,
        run_correction=True, correction_tolerance_km=0.05,
    )

    loose_result = run_experiment(loose_config)
    tight_result = run_experiment(tight_config)

    assert loose_result.maneuver_delta_v_km_s is not None
    assert tight_result.maneuver_delta_v_km_s is not None
    assert tight_result.maneuver_delta_v_km_s >= loose_result.maneuver_delta_v_km_s - 1e-6
