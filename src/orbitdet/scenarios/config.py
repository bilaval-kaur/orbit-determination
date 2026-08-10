"""
Scenario configuration schema.

A ScenarioConfig fully specifies one reproducible run: the initial truth
state, the gravitational environment, propagation timing, and the random
seed. Every experiment in this project is defined as a YAML file that
loads into this schema - nothing about a run's conditions should ever
live as a hardcoded value inside a module.

Units and frame conventions are fixed project-wide in docs/conventions.md.
This schema enforces them at the boundary: if a config violates a
convention (wrong vector length, non-positive duration, etc.), it fails
loudly here, before any physics code runs on bad input.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator, model_validator

# Earth's gravitational parameter, km^3/s^2.
# Source: Vallado, "Fundamentals of Astrodynamics and Applications" (5th ed.),
# Appendix D (Earth physical/geophysical constants) - this is the standard
# tabulated value used throughout the astrodynamics literature. We default
# to it but allow override, since some validation cases (e.g. cross-checking
# against a textbook worked example) may specify a slightly different
# published value.
MU_EARTH_KM3_S2 = 398600.4418


class ScenarioConfig(BaseModel):
    """
    Fully specifies one reproducible simulation scenario.

    All positions are in km, all velocities in km/s, all times in seconds
    unless explicitly named otherwise (e.g. epoch_utc). The initial state
    is expressed in an idealized Earth-centered inertial (ECI) frame - see
    docs/conventions.md for exactly what that does and doesn't mean at
    this stage of the project.
    """

    name: str = Field(..., description="Short identifier for this scenario, used in output filenames.")
    seed: int = Field(..., description="Random seed for all stochastic elements (measurement noise, etc.).")

    epoch_utc: datetime = Field(
        ..., description="Reference epoch (t=0) for this scenario, as a UTC timestamp."
    )

    mu_km3_s2: float = Field(
        default=MU_EARTH_KM3_S2,
        gt=0,
        description="Gravitational parameter of the central body, km^3/s^2.",
    )

    initial_position_km: tuple[float, float, float] = Field(
        ..., description="Initial position vector [x, y, z] in the ECI frame, km."
    )
    initial_velocity_km_s: tuple[float, float, float] = Field(
        ..., description="Initial velocity vector [vx, vy, vz] in the ECI frame, km/s."
    )

    propagation_duration_s: float = Field(
        ..., gt=0, description="Total simulated time to propagate, seconds."
    )
    integrator_step_s: float = Field(
        ..., gt=0, description="Fixed integrator step size, seconds."
    )

    @field_validator("epoch_utc")
    @classmethod
    def epoch_must_be_utc(cls, v: datetime) -> datetime:
        """
        Reject naive or non-UTC datetimes. A silently-assumed timezone is
        exactly the kind of ambiguity this schema exists to eliminate.
        """
        if v.tzinfo is None:
            raise ValueError("epoch_utc must be timezone-aware (include a UTC offset, e.g. '...Z' or '+00:00').")
        if v.utcoffset().total_seconds() != 0:
            raise ValueError("epoch_utc must be in UTC (offset +00:00), not another timezone.")
        return v

    @model_validator(mode="after")
    def step_must_not_exceed_duration(self) -> "ScenarioConfig":
        """
        A step size larger than the total duration would silently produce
        zero or one propagated points - not a crash, just a meaningless
        result. Catch it here instead of downstream.
        """
        if self.integrator_step_s > self.propagation_duration_s:
            raise ValueError(
                f"integrator_step_s ({self.integrator_step_s}) must not exceed "
                f"propagation_duration_s ({self.propagation_duration_s})."
            )
        return self


def load_scenario_config(path: str) -> ScenarioConfig:
    """
    Load and validate a scenario configuration from a YAML file.

    Raises pydantic.ValidationError with a clear, field-specific message
    if the file is malformed or violates a project convention.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return ScenarioConfig(**raw)
