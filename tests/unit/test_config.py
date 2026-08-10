"""
Unit tests for ScenarioConfig.

These protect against the exact failure mode M0 exists to prevent: a
scenario config that looks reasonable but silently violates a project
convention (wrong timezone, non-physical duration/step relationship,
wrong-length vectors) getting past validation and corrupting a
downstream physics result.
"""

import pytest
from pydantic import ValidationError

from orbitdet.scenarios.config import ScenarioConfig, load_scenario_config


def _valid_kwargs() -> dict:
    """A minimal, valid set of config fields, reused and overridden per-test."""
    return dict(
        name="test_scenario",
        seed=1,
        epoch_utc="2026-01-01T00:00:00+00:00",
        initial_position_km=(6778.137, 0.0, 0.0),
        initial_velocity_km_s=(0.0, 7.6685582, 0.0),
        propagation_duration_s=18000.0,
        integrator_step_s=10.0,
    )


def test_valid_config_loads_successfully():
    config = ScenarioConfig(**_valid_kwargs())
    assert config.name == "test_scenario"
    assert config.mu_km3_s2 == pytest.approx(398600.4418)


def test_mu_defaults_to_earth_when_not_specified():
    config = ScenarioConfig(**_valid_kwargs())
    assert config.mu_km3_s2 == pytest.approx(398600.4418)


def test_naive_datetime_is_rejected():
    """A timezone-naive epoch is exactly the kind of silent ambiguity this schema must not allow."""
    kwargs = _valid_kwargs()
    kwargs["epoch_utc"] = "2026-01-01T00:00:00"
    with pytest.raises(ValidationError):
        ScenarioConfig(**kwargs)


def test_non_utc_timezone_is_rejected():
    kwargs = _valid_kwargs()
    kwargs["epoch_utc"] = "2026-01-01T00:00:00+05:30"
    with pytest.raises(ValidationError):
        ScenarioConfig(**kwargs)


def test_step_larger_than_duration_is_rejected():
    kwargs = _valid_kwargs()
    kwargs["integrator_step_s"] = 99999.0
    kwargs["propagation_duration_s"] = 100.0
    with pytest.raises(ValidationError):
        ScenarioConfig(**kwargs)


def test_negative_duration_is_rejected():
    kwargs = _valid_kwargs()
    kwargs["propagation_duration_s"] = -10.0
    with pytest.raises(ValidationError):
        ScenarioConfig(**kwargs)


def test_wrong_length_position_vector_is_rejected():
    kwargs = _valid_kwargs()
    kwargs["initial_position_km"] = (6778.137, 0.0)
    with pytest.raises(ValidationError):
        ScenarioConfig(**kwargs)


def test_load_scenario_config_reads_yaml_file(tmp_path):
    """Integration-style check: the actual nominal_two_body.yaml shape loads correctly end to end."""
    yaml_content = """
name: "yaml_test"
seed: 7
epoch_utc: "2026-01-01T00:00:00+00:00"
initial_position_km: [6778.137, 0.0, 0.0]
initial_velocity_km_s: [0.0, 7.6685582, 0.0]
propagation_duration_s: 18000.0
integrator_step_s: 10.0
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    config = load_scenario_config(str(config_file))
    assert config.name == "yaml_test"
    assert config.seed == 7
