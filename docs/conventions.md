# Conventions

This document is the single source of truth for units, frames, and numerical
conventions used throughout this project. Every module must follow it without
exception. When in doubt, this file wins — if code and this document
disagree, the code has a bug.

Read this before writing or reviewing any physics code. Most orbital
mechanics bugs are not math errors — they are unit or frame mismatches that
run without crashing and produce confidently wrong answers.

---

## 1. Units

| Quantity | Unit | Notes |
|---|---|---|
| Length / position | **kilometers (km)** | Not meters. Standard in astrodynamics. |
| Velocity | **km/s** | Follows from the length convention. |
| Time (durations, step sizes) | **seconds (s)** | All propagation and integration quantities. |
| Time (epoch / absolute) | **UTC timestamp** (ISO 8601, timezone-aware) | Converted to elapsed seconds internally. |
| Angles (internal math) | **radians** | All trigonometric functions, all internal storage. |
| Angles (config files, human-readable output) | **degrees** | Converted to radians immediately on load. |
| Gravitational parameter (mu) | **km^3/s^2** | Consistent with km and s above. |
| Mass-specific energy (epsilon) | **km^2/s^2** | Per unit mass; no separate mass unit in this project. |
| Specific angular momentum (h) | **km^2/s** | |

Rule: any function boundary that accepts or returns a bare float for a
physical quantity must name the unit in the variable name or type hint
(altitude_km, not altitude). Same convention as Mission Control.

## 2. Reference frame

All state vectors (position and velocity) are expressed in an idealized
Earth-Centered Inertial (ECI) frame: origin at Earth's center, axes fixed
relative to the stars (not rotating with Earth), right-handed.

Important limitation to state honestly: at the two-body stage of this
project (M1-M10), this frame is NOT tied to a specific real-world frame
realization (e.g., GCRF/J2000, which has a precisely defined orientation via
IAU conventions and is what real operational systems use). Pure two-body
motion has no physical mechanism that cares about the frame's exact
orientation relative to the stars - any fixed inertial frame gives identical
dynamics. We use "ECI" here as a conceptual label (origin at Earth's
center, inertial, right-handed), not a claim of alignment with a specific
real astronomical frame.

This stops being a free simplification the moment J2 is introduced
(stretch goal S1). J2 perturbation formulas assume the frame's z-axis is
aligned with Earth's actual rotation axis - because J2 physically models
Earth's equatorial bulge, which only makes sense relative to Earth's real
equator. If S1 is implemented, this section must be revisited and the frame
assumption made explicit and physically grounded. Flagging this now so it
isn't a silent correctness gap later.

## 3. Time handling

- epoch_utc in a scenario config is a human-readable label for t=0 - a
  real UTC calendar timestamp, required to be timezone-aware and in UTC
  specifically (see ScenarioConfig validators).
- All propagation internally uses elapsed seconds since epoch, t,
  starting at t=0. Two-body dynamics under the Keplerian assumption have no
  explicit dependence on absolute calendar time - only elapsed time matters.
- If a future module needs to convert back to a calendar timestamp (e.g., for
  a plot axis), do that conversion at the presentation boundary, not inside
  any dynamics or estimation code.

## 4. State vector convention

The state vector is:

x = [rx, ry, rz, vx, vy, vz]

- Position components first, then velocity - this ordering is fixed
  project-wide. Every Jacobian, every covariance matrix row/column, every
  measurement model assumes this order.
- Stored in code as a NumPy array of shape (6,), not (6, 1), unless a
  specific linear-algebra operation requires an explicit column vector (e.g.,
  a matrix-vector product where shape matters) - in that case, reshape
  locally and reshape back; don't let (6, 1) arrays leak across module
  boundaries, since they silently break broadcasting in ways that are
  painful to debug.
- Covariance matrices are (6, 6), with row/column order matching the state
  vector order above exactly.

## 5. Why there is no mass in this project's units

Notice mu, epsilon, and h are all "per unit mass" quantities. This isn't a
simplification we're choosing - it's a structural fact about the two-body
problem: a test particle's orbit around a much larger central body does not
depend on the test particle's own mass at all (this is the same physical
fact as "a feather and a hammer fall at the same rate," extended to orbits).
Spacecraft mass only re-enters the picture when converting a delta-v (velocity
change, km/s - what we compute) into a propellant mass via the rocket
equation - which is out of scope for this project unless explicitly added as
a later extension. Every delta-v value in this project is a velocity, never a
mass or a force.

## 6. Random seeds and reproducibility

- Every ScenarioConfig requires an explicit integer seed.
- All stochastic elements (measurement noise, Monte Carlo trials) must be
  generated from a numpy.random.Generator seeded from this value -
  np.random.default_rng(seed) - never from NumPy's legacy global random
  state (np.random.seed(...) / np.random.randn(...)), which is harder to
  reason about and not safe if multiple components need independent random
  streams.
- Two runs with the same config file must produce bit-identical results.
  This is what makes an "experiment" in this project actually an experiment,
  rather than an anecdote.

## 7. Numerical conventions

- Fixed-step RK4 is the default integrator for the truth engine (M1) -
  simple, well-understood, adequate accuracy for the timescales here. Its
  limitations (fixed truncation error per step, no adaptive step-size
  control) are validated explicitly, not assumed away - see M1's
  energy/momentum conservation tests.
- scipy.integrate.solve_ivp with method DOP853 (high-order, adaptive) is
  used ONLY as an independent cross-validation reference for the
  hand-written RK4 - never as the primary propagator. If the two ever
  disagree beyond numerical tolerance, trust DOP853 and go find the bug in
  the RK4 implementation or its inputs.

## 8. Logging

Use Python's standard logging module (as in Mission Control), not print.
Log at decision points and failure points (e.g., "NIS consistency test
failed," "optimizer did not converge") - not on every function call.
