# Autonomous Satellite Orbit Determination & Maneuver Planning

**Given imperfect observations of a spacecraft, can we estimate its state,
detect orbital deviation, and autonomously determine an efficient
corrective maneuver?**

A from-scratch implementation of orbit determination, statistical filter
consistency validation, anomaly detection, and constrained minimum-fuel
maneuver optimization -- built to demonstrate estimation theory, numerical
methods, and operations-research thinking applied to a real aerospace
problem, not to produce a toy simulator.

---

## Overview

This project simulates a spacecraft in low Earth orbit, generates realistic
noisy measurements of its position, estimates its true state using a
hand-derived Extended Kalman Filter, formally validates that the filter's
self-reported uncertainty is statistically honest, detects real unmodeled
orbital deviations via principled hypothesis testing, and plans
minimum-fuel corrective maneuvers via constrained nonlinear optimization --
all built on physics and mathematics derived from first principles, not
assumed from a library.

## Motivation

I am a Chemical Engineering graduate pursuing graduate study in Aerospace
Engineering, Systems Engineering, and Operations Research. This project
exists to demonstrate that transition is deliberate and technically
grounded -- not a claim, a working artifact. It deliberately differs from
my other satellite project,
[Mission Control](https://github.com/bilaval-kaur/mission-control): that
project demonstrates full-stack systems engineering built on a trusted,
industry-standard propagator (SGP4/Skyfield). This project derives the
dynamics, the filter, and the optimization myself -- almost nothing here is
blackboxed except the ODE solver's internal step-control and the
optimizer's internal search algorithm, both of which I supply the model,
Jacobian, and constraints for.

## Features

- **Real two-body orbital dynamics**, hand-derived from Newton's law of
  gravitation, integrated with a hand-written RK4 propagator, validated
  against energy/angular-momentum conservation and cross-checked against
  `scipy.solve_ivp`
- **Orbital elements conversion**, correctly handling circular/equatorial
  geometric singularities
- **Reproducible measurement simulation** with seeded Gaussian noise
- **A hand-built Extended Kalman Filter**, including an analytically
  derived state transition Jacobian, cross-validated against finite
  differences
- **Formal filter consistency validation** (NEES/NIS, chi-squared
  hypothesis testing) -- not just "does the estimate look accurate," but
  "is the filter's uncertainty honest"
- **Statistical anomaly detection** via single-sample NIS hypothesis
  testing with M-of-N persistence logic, catching a real injected
  perturbation within ~30 seconds
- **Physically-grounded impulsive maneuver modeling** (RTN frame), with a
  numerically-proven efficiency argument for why plane changes are
  expensive
- **Constrained minimum-fuel maneuver optimization** via `scipy.optimize`
  (SLSQP), replacing an originally-proposed but flawed weighted-sum
  objective with the standard aerospace formulation
- **A config-driven experiment framework** running 7 controlled
  experiments, with results independently reproducing earlier findings
  through an entirely different code path
- **54+ automated tests** spanning unit, physical-validation, and
  integration categories

## System Architecture

    Scenario config (ExperimentConfig)
            |
            v
    Truth Engine (two-body dynamics + RK4)  --never seen by the filter--+
            |                                                            |
            v                                                            |
    Measurement Simulator (adds seeded Gaussian noise)                   |
            |                                                            |
            v                                                            |
    Estimator (EKF: predict + update)  <--only sees measurements---------+
            |
            v
    Consistency Validator (NEES/NIS, chi-squared)
            |
            v
    Anomaly Detector (single-sample + persistence)
            |
            v
    Maneuver Planner (RTN burn model)
            |
            v
    Optimizer (SLSQP, constrained minimum-fuel)
            |
            v
    Metrics + Visualization

The estimator never sees ground truth directly -- only noisy measurements --
enforced structurally throughout the codebase, since this is the entire
epistemic basis of the project.

## Technology Stack

Python 3.12+, NumPy, SciPy (`optimize`, `integrate`, `stats`), Pandas,
Matplotlib, Pydantic (config schema validation), PyYAML, pytest, Git.

## Mathematics

Every derivation used in this project -- the two-body equation of motion,
conserved-quantity proofs, the EKF's Jacobian derivation, the chi-squared
consistency theory, the RTN maneuver efficiency proof, and the constrained
optimization formulation -- is documented in full in
[`docs/mathematics.md`](docs/mathematics.md).

## Data Honesty

Following the same principle as Mission Control: this project is entirely
**simulated** -- there is no real spacecraft telemetry anywhere in this
repository, and it does not claim to be. What *is* real: the physics (the
two-body equation of motion, SGP4-independent orbital mechanics), the
mathematics (every derivation is genuine, checkable astrodynamics and
estimation theory), and the numerical results (every number reported was
actually computed by running the code, not fabricated). What's synthetic:
the specific scenarios, noise realizations, and injected anomalies, all
generated from explicitly seeded random number generators for full
reproducibility.

## Installation

Prerequisites: Python 3.12+, Git.

    git clone https://github.com/bilaval-kaur/orbit-determination.git
    cd orbit-determination
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Running

Run the full test suite:

    pytest tests/ -v

Run all 7 controlled experiments and produce a summary table:

    python3 experiments/run_all_experiments.py

Generate the core visualization figures:

    python3 experiments/generate_visualizations.py

## Repository Structure

    src/orbitdet/
      dynamics/       two-body EOM, RK4 integrator
      elements/       orbital elements <-> state vector conversion
      measurements/   noisy measurement simulation
      estimation/     Extended Kalman Filter, NEES/NIS consistency
      detection/      statistical anomaly detection
      maneuver/       impulsive maneuver model, RTN frame
      optimization/   minimum-fuel constrained optimization
      scenarios/      config schema, experiment runner
    configs/          YAML scenario definitions
    experiments/      experiment + visualization scripts, results
    tests/
      unit/                     component-level tests
      integration/              cross-component tests (e.g. RK4 vs solve_ivp)
      physical_validation/      conservation laws, physical constraints
    docs/
      conventions.md            units, frames, numerical conventions
      mathematics.md            full derivations
      figures/                  generated visualizations

## Testing

Tests are split by what they validate, not just what they cover:

- **Unit tests** -- does one function do what it claims, in isolation
- **Physical validation** -- does the system obey real physics (energy/
  momentum conservation, valid orbital elements, filter consistency)
- **Integration tests** -- do independently-implemented components agree
  (hand-written RK4 vs. `scipy.solve_ivp`; analytical Jacobian vs. finite
  differences)

Run with `pytest tests/ -v`.

## Limitations

- Visibility/measurement model is direct position only -- range/range-rate
  (a nonlinear measurement requiring a real Jacobian) is a documented
  stretch goal, not implemented
- NEES/NIS consistency intervals use time-samples from a single trajectory
  as an approximation to independent trials; a fully rigorous test requires
  Monte Carlo averaging across independent seeds (see Future Improvements)
- The reference frame is an idealized inertial frame, not tied to a real
  astronomical realization (GCRF/J2000) -- a free simplification for pure
  two-body dynamics that would need revisiting if J2 were added
- A real, documented systems-level finding: the process noise `Q` tuned
  for nominal filter consistency (M5) makes the filter slow to adapt after
  a genuine anomaly occurs (M9) -- a real tension in estimation theory
  between calibration and adaptability, not resolved in this version
- Anomaly injection and maneuver planning support one active event at a
  time; concurrent/overlapping anomalies are not modeled
- Maneuver "cost" is Delta-v (velocity), never propellant mass -- converting
  to mass via the rocket equation is out of scope

## Future Improvements

- J2 perturbation in the truth model with a deliberate truth/filter model
  mismatch, to study and quantify the resulting filter bias
- Monte Carlo campaigns (independent trial averaging) for fully rigorous
  NEES/NIS consistency testing, replacing the single-trajectory
  approximation
- Range/range-rate measurement model with a properly derived nonlinear
  Jacobian
- Adaptive process noise / covariance inflation triggered on anomaly
  detection, to resolve the detection-vs-adaptation tension documented in
  Limitations
- Two-burn correction strategies
- Direct-transcription trajectory optimization

## Engineering Challenges

A few genuine findings worth highlighting specifically, each diagnosed
through real debugging rather than assumed correct:

1. **A dimensional-consistency bug** in orbital element singularity
   detection (comparing a dimensional quantity to a fixed absolute
   threshold, rather than a scale-invariant ratio).
2. **An EKF overconfidence discovery**: a near-zero process noise `Q`
   produced NEES nearly double its theoretical mean, traced to
   linearization error in covariance propagation being unabsorbed -- and a
   single-seed tuning that looked correct but failed under multi-seed
   averaging, directly motivating the project's methodological discipline
   around sampling.
3. **An ill-conditioned optimization landscape**, where a ~0.001 km/s
   change in a candidate maneuver caused an 11x swing in resulting
   position error -- correctly diagnosed as a real property of orbital
   dynamics amplifying small velocity differences, and the rigorous
   argument for why gradient-based optimization is the right primary tool
   over blind search.
4. **A target-time mismatch bug** in the visualization pipeline that
   produced a physically absurd multi-km/s "correction" -- traced to a
   1000-second discrepancy between a target's reference time and the
   correction's actual propagation window.

## What I Learned

This project required genuinely deriving -- not just calling library
functions for -- orbital dynamics, coordinate transformations, Kalman
filter theory, chi-squared consistency testing, and constrained
optimization. It also required real engineering judgment: correcting a
flawed objective function proposal before writing any code, catching
methodologically unsound single-seed conclusions before they became false
confidence, and documenting honest limitations rather than hiding them. The
recurring lesson, across nearly every milestone: verifying an assumption
empirically is worth more than trusting it looks right.

## License

MIT License.
