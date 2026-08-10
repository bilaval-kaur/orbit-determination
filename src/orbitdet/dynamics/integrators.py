"""
Fixed-step RK4 integrator.

This is the project's primary (hand-written) propagator -- see
docs/conventions.md Section 7 for why RK4, and why scipy.solve_ivp
(DOP853) is used only as an independent cross-validation reference,
never as the primary engine.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def rk4_step(
    f: Callable[[float, np.ndarray, float], np.ndarray],
    t: float,
    state: np.ndarray,
    h: float,
    mu: float,
) -> np.ndarray:
    """
    Advance one RK4 step.

    Evaluates the derivative function f four times -- at the start of
    the step, twice at the midpoint (using progressively better
    estimates of the state there), and once at the end -- then combines
    them as a weighted average to estimate the state after time h.
    Local truncation error O(h^5), global error O(h^4).

    Args:
        f: derivative function, signature f(t, state, mu) -> derivative.
        t: current time, seconds.
        state: current (6,) state vector.
        h: step size, seconds.
        mu: gravitational parameter, km^3/s^2 (passed through to f).

    Returns:
        (6,) state vector after advancing by h.
    """
    k1 = f(t, state, mu)
    k2 = f(t + h / 2.0, state + h / 2.0 * k1, mu)
    k3 = f(t + h / 2.0, state + h / 2.0 * k2, mu)
    k4 = f(t + h, state + h * k3, mu)

    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_rk4(
    f: Callable[[float, np.ndarray, float], np.ndarray],
    t0: float,
    state0: np.ndarray,
    step_s: float,
    duration_s: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate a state forward over a fixed duration using fixed-step RK4.

    Args:
        f: derivative function, signature f(t, state, mu) -> derivative.
        t0: initial time, seconds.
        state0: initial (6,) state vector.
        step_s: fixed integrator step size, seconds.
        duration_s: total time to propagate, seconds.
        mu: gravitational parameter, km^3/s^2.

    Returns:
        (times, states): times is a (N,) array, states is a (N, 6) array,
        where N = floor(duration_s / step_s) + 1 (including t0).
    """
    n_steps = int(np.floor(duration_s / step_s))

    times = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, 6))

    times[0] = t0
    states[0] = state0

    t = t0
    state = state0.copy()
    for i in range(1, n_steps + 1):
        state = rk4_step(f, t, state, step_s, mu)
        t += step_s
        times[i] = t
        states[i] = state

    return times, states
