"""
Tests for the measurement simulator.

Protects against: non-reproducible randomness (violating
docs/conventions.md Section 6), incorrect noise statistics, and
incorrect H matrix / shape handling.
"""

import numpy as np
import pytest

from orbitdet.measurements.models import (
    generate_measurement_series,
    position_measurement_matrix,
    simulate_position_measurement,
)

TRUE_STATE = np.array([6778.137, 0.0, 0.0, 0.0, 7.6685582, 0.0])


def test_position_measurement_matrix_has_correct_shape_and_structure():
    H = position_measurement_matrix()
    assert H.shape == (3, 6)
    np.testing.assert_array_equal(H[:, 0:3], np.eye(3))
    np.testing.assert_array_equal(H[:, 3:6], np.zeros((3, 3)))


def test_measurement_matrix_correctly_extracts_position():
    """H @ state should equal exactly the position components, with zero noise involved."""
    H = position_measurement_matrix()
    extracted = H @ TRUE_STATE
    np.testing.assert_array_equal(extracted, TRUE_STATE[0:3])


def test_same_seed_produces_identical_measurements():
    """
    Reproducibility is a hard project requirement (docs/conventions.md
    Section 6) -- two Generators seeded identically must produce
    bit-identical noise.
    """
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    m1 = simulate_position_measurement(TRUE_STATE, sigma_km=0.5, rng=rng1)
    m2 = simulate_position_measurement(TRUE_STATE, sigma_km=0.5, rng=rng2)

    np.testing.assert_array_equal(m1, m2)


def test_different_seeds_produce_different_measurements():
    """Sanity check the opposite direction: different seeds must NOT coincidentally agree."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)

    m1 = simulate_position_measurement(TRUE_STATE, sigma_km=0.5, rng=rng1)
    m2 = simulate_position_measurement(TRUE_STATE, sigma_km=0.5, rng=rng2)

    assert not np.array_equal(m1, m2)


def test_measurement_noise_statistics_match_configured_sigma():
    """
    Statistical validation: over many samples, the empirical standard
    deviation of the noise should converge close to the configured
    sigma_km. This is checking the underlying Gaussian generator is
    actually being used with the right scale, not just "some noise".
    """
    rng = np.random.default_rng(123)
    sigma_km = 0.5
    n_samples = 20000

    samples = np.array(
        [simulate_position_measurement(TRUE_STATE, sigma_km, rng) for _ in range(n_samples)]
    )
    noise = samples - TRUE_STATE[0:3]

    empirical_std = np.std(noise, axis=0)
    # 5% relative tolerance is generous enough to avoid a flaky test from
    # sampling variation at 20000 samples, while still catching a real
    # scale error (e.g. sigma used where sigma^2 was intended, or vice versa).
    np.testing.assert_allclose(empirical_std, sigma_km, rtol=0.05)

    empirical_mean = np.mean(noise, axis=0)
    # Mean should be close to zero (unbiased noise) -- absolute, not
    # relative, tolerance here since the true mean is exactly zero.
    np.testing.assert_allclose(empirical_mean, 0.0, atol=0.02)


def test_generate_measurement_series_shape_matches_input_trajectory():
    n_points = 50
    fake_trajectory = np.tile(TRUE_STATE, (n_points, 1))  # (50, 6), same state repeated

    rng = np.random.default_rng(7)
    measurements = generate_measurement_series(fake_trajectory, sigma_km=0.5, rng=rng)

    assert measurements.shape == (n_points, 3)


def test_generate_measurement_series_is_reproducible_with_same_seed():
    n_points = 20
    fake_trajectory = np.tile(TRUE_STATE, (n_points, 1))

    series1 = generate_measurement_series(fake_trajectory, sigma_km=0.3, rng=np.random.default_rng(99))
    series2 = generate_measurement_series(fake_trajectory, sigma_km=0.3, rng=np.random.default_rng(99))

    np.testing.assert_array_equal(series1, series2)
