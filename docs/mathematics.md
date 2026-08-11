# Mathematics

This document consolidates every derivation used in this project, referenced
throughout the codebase as `docs/mathematics.md`. Each section states the
result, the derivation, and the physical or statistical meaning -- not just
the equation.

---

## 1. Two-Body Dynamics

### 1.1 The equation of motion

Two point masses under mutual gravitation reduce to a single relative-motion
equation. Newton's law of gravitation on each body:m1 R1_ddot = G m1 m2 (R2-R1)/|R2-R1|^3
m2 R2_ddot = -G m1 m2 (R2-R1)/|R2-R1|^3

Define `r = R2 - R1` (spacecraft relative to central body). Subtracting:
r_ddot = -G(m1+m2) r/|r|^3

Define `mu = G(m1+m2) ~= G*m1` (spacecraft mass is negligible next to
Earth's, `m2/m1 ~ 10^-22`). Final form:
r_ddot = -mu * r / |r|^3

This reduction is **exact**; only dropping the spacecraft's mass from `mu`
is an approximation (an extremely good one).

### 1.2 First-order state-space form

State `x = [r, v]` (6,). The system becomes:
r_dot = v
v_dot = -mu * r / |r|^3

Implemented in `dynamics/two_body.py::two_body_eom`.

### 1.3 Conserved quantities

**Specific orbital energy**: `epsilon = v^2/2 - mu/r`. Proof of conservation:
d/dt(v^2/2) = v . v_dot = -mu(v.r)/r^3
d/dt(-mu/r) = mu(r.v)/r^3 [using d|r|/dt = (r.v)/|r|]
sum = 0

**Specific angular momentum**: `h = r x v`. Proof:
d/dt(r x v) = v x v + r x (-mu r/r^3) = 0 + 0

`h` is conserved as a full vector (magnitude AND direction) -- this is why
orbits are planar: `r` stays perpendicular to the fixed vector `h`.

### 1.4 Assumptions

Point-mass central body (no J2), no drag, no third bodies, no relativistic
effects, inertial non-rotating frame. See `docs/conventions.md` for the full
list and the frame-realization caveat.

---

## 2. Numerical Integration (RK4)

Fourth-order Runge-Kutta, one step of size `h`:
k1 = f(t, x)
k2 = f(t + h/2, x + h/2 * k1)
k3 = f(t + h/2, x + h/2 * k2)
k4 = f(t + h, x + h * k3)
x_new = x + h/6 * (k1 + 2k2 + 2k3 + k4)

Local truncation error `O(h^5)`, global error `O(h^4)`. Cross-validated
against `scipy.integrate.solve_ivp` (DOP853) in `tests/integration/`.


---

## 3. Orbital Elements

Classical (Keplerian) elements (a, e, i, RAAN, argp, nu) fully describe an
orbit, equivalent to the state vector. Conversion algorithm
(elements/conversions.py):

r, v magnitudes
v_radial = (r.v)/r
h = r x v
i = arccos(h_z / |h|)
N = z_hat x h              (node vector)
RAAN = arccos(N_x/|N|), quadrant-corrected by N_y
e_vec = (1/mu)[(v^2 - mu/r) r - (r.v) v]     (eccentricity vector, from the
                                               Laplace-Runge-Lenz vector)
argp = arccos((N.e_vec)/(|N||e|)), quadrant-corrected by e_vec_z
nu = arccos((e_vec.r)/(|e|r)), quadrant-corrected by v_radial
a = -mu / (2*epsilon)        (direct reuse of Section 1.3's conserved energy)

Singularities: circular orbits (e~0) leave argp/nu undefined; equatorial
orbits (i~0 or 180) leave RAAN undefined (node vector is zero). Handled via
documented fallback conventions and is_circular/is_equatorial flags rather
than dividing by near-zero magnitudes.

Inverse conversion (perifocal frame + 3-1-3 Euler rotation):

p = a(1-e^2)                              (semi-latus rectum)
r_pf = (p/(1+e cos(nu))) [cos(nu), sin(nu), 0]
v_pf = (mu/h) [-sin(nu), e+cos(nu), 0]
R = Rz(RAAN) @ Rx(i) @ Rz(argp)
r_eci = R @ r_pf,  v_eci = R @ v_pf

---

## 4. Measurement Model

Direct position measurement, linear:

z = H x + v,   v ~ N(0, R)
H = [I_3 | 0_3]              (3x6, selects position components)

Isotropic Gaussian noise, drawn from a seeded numpy.random.Generator (see
docs/conventions.md Section 6).

---

## 5. Extended Kalman Filter

### 5.1 Predict

x_pred = f(x_prev)                        (RK4 propagation, Section 2)
P_pred = Phi @ P_prev @ Phi.T + Q

Phi (state transition matrix) is propagated by integrating the variational
equations Phi_dot = A(x) @ Phi, Phi(0)=I, alongside the state, using the
same RK4 machinery as Section 2.

### 5.2 The dynamics Jacobian (why it's "Extended")

A = df/dx, block form:

A = [ 0_3   I_3 ]
    [ G     0_3 ]

G = da/dr, derived by direct differentiation of a = -mu r/r^3:

d(a_i)/d(r_j) = -mu[delta_ij/r^3 - 3 r_i r_j/r^5]
G = -mu/r^3 * (I_3 - 3 r_hat r_hat^T)

Cross-validated against a finite-difference Jacobian in
tests/unit/test_ekf.py.

### 5.3 Update

innovation (nu) = z - H x_pred
S = H P_pred H^T + R
K = P_pred H^T S^-1
x_new = x_pred + K nu
P_new = (I-KH) P_pred (I-KH)^T + K R K^T      (Joseph form)

Kalman gain intuition: K optimally blends prediction and measurement by
their relative uncertainty -- large R (noisy sensor) shrinks K (trust the
model); large P (uncertain model) grows K (trust the sensor). Joseph form
guarantees P stays symmetric/PSD under floating-point roundoff, unlike the
simpler (I-KH)P form.

### 5.4 Process noise Q -- why it isn't near-zero

Even with a physically perfect dynamics model, P's propagation
(Phi P Phi^T) is a linear approximation of uncertainty propagating through
nonlinear dynamics. That linearization loses information every step. With
Q too small, nothing absorbs this loss and the filter becomes overconfident
(see Section 6). Q is tuned, not derived from first principles, to
compensate for this -- a standard, documented practice (see Bar-Shalom,
Estimation with Applications to Tracking and Navigation).

---

## 6. Filter Consistency: NEES, NIS, and the Chi-Squared Distribution

### 6.1 The chi-squared distribution

If Z_1...Z_k are independent standard normal, X = sum(Z_i^2) ~ chi2(k).
Mean k, variance 2k.

Why quadratic error forms are chi-squared: if e ~ N(0, P), decompose
P = LL^T (Cholesky). Then w = L^-1 e is standard Gaussian (independent
unit-variance components), and:

e^T P^-1 e = e^T (LL^T)^-1 e = (L^-1 e)^T (L^-1 e) = w^T w = sum(w_i^2) ~ chi2(n)

### 6.2 NEES and NIS

NEES_k = e_k^T P_k^-1 e_k,   e_k = x_true - x_hat      (requires ground truth)
NIS_k  = nu_k^T S_k^-1 nu_k                              (filter-only, no truth needed)

Consistent filter: E[NEES] = n (state dim, 6), E[NIS] = m (measurement
dim, 3).

### 6.3 Consistency interval

For N independent samples, N * NEES_bar ~ chi2(N*n), giving a (1-alpha)
interval [chi2_(a/2)(Nn)/N, chi2_(1-a/2)(Nn)/N].

Documented limitation: this project's implementation uses TIME samples
from a single trajectory as an approximation to independent trials (true
independence requires Monte Carlo across seeds -- stretch goal S2).
Empirically, single-seed tuning proved unreliable (M5's development found a
Q that passed on one seed but failed averaged across five) -- multi-seed
averaging is used wherever consistency is formally asserted.

### 6.4 Single-sample anomaly detection

One-sided test: NIS_k > chi2_(1-alpha)(m) flags a single measurement.
One-sided because a real dynamical deviation only increases NIS; nothing
in this project's measurement model produces suspiciously low NIS as a
fault signature. Even a perfectly consistent filter flags at rate alpha
by chance -- M-of-N persistence logic (requiring several flags within a
recent window) suppresses this while preserving fast detection of genuine,
sustained deviations.

---

## 7. Impulsive Maneuvers

### 7.1 The impulsive assumption

r(t+) = r(t-)              (position continuous)
v(t+) = v(t-) + delta_v    (velocity jumps)

Valid when burn duration << orbital period.

### 7.2 The RTN frame

R_hat = r/|r|                    (radial, outward)
N_hat = (r x v)/|r x v|          (normal, along angular momentum)
T_hat = N_hat x R_hat             (transverse, completes right-hand frame)

### 7.3 Tangential vs. normal burn efficiency (proof)

For small delta_v:

|v+dv|^2 = |v|^2 + 2(v.dv) + |dv|^2

Tangential dv (aligned with v): v.dv large -> energy changes at first
order in dv.

Normal dv (perpendicular to v): v.dv = 0 exactly -> only |dv|^2 survives
-> energy changes at second order in dv.

For small dv, first-order dominates second-order -- empirically confirmed
in tests/unit/test_maneuver.py: at dv=0.001 km/s, the tangential/normal
energy-change ratio exceeds 15,000x. This is the rigorous basis for "plane
changes are expensive."

---

## 8. Minimum-Fuel Maneuver Optimization

### 8.1 Formulation

Replaces an originally-proposed weighted-sum objective (w1*error +
w2*dv + w3*time), rejected for combining incompatible units without a
principled way to choose weights. Standard aerospace formulation instead:

minimize    ||delta_v||
subject to  position_error(delta_v) <= tolerance

### 8.2 Smoothing

Minimizes ||delta_v||^2 (smooth everywhere) rather than ||delta_v||
(non-smooth at zero) -- since sqrt is monotonic for non-negative reals,
both share the same minimizer; the true delta_v magnitude is reported as
the final cost.

### 8.3 Solved via SLSQP

scipy.optimize.minimize, Sequential Least Squares Programming, with one
inequality constraint g(x) = tolerance - position_error(x) >= 0. A
fundamental property, tested directly: tightening the constraint (smaller
tolerance) can never decrease the optimal objective, only increase or
maintain it (a smaller feasible region cannot contain a better minimum than
a larger one that contained it).

### 8.4 A discovered ill-conditioning

For some scenarios, the position-error-vs-delta_v landscape near the
optimum is extremely sensitive (a ~0.001 km/s change producing an 11x swing
in resulting error), because dynamics propagated over a meaningful fraction
of an orbital period amplify small velocity differences -- the same
sensitivity that motivates the EKF's state transition matrix (Section 5.2).
This is precisely why gradient-based optimization (SLSQP) is the right
primary tool here, not blind/grid search: a gradient method follows the
local slope directly to a narrow, steep optimum that a grid can step over
entirely.

---

## 9. References

- Vallado, Fundamentals of Astrodynamics and Applications, 5th ed. (2022)
  -- orbital maneuvering, orbit determination.
- Curtis, Orbital Mechanics for Engineering Students, 4th ed. (2019) --
  two-body dynamics, orbital elements.
- Tapley, Schutz & Born, Statistical Orbit Determination (2004) --
  sequential filtering, covariance analysis.
- Bar-Shalom, Li & Kirubarajan, Estimation with Applications to Tracking
  and Navigation (2001) -- Kalman filter theory, NEES/NIS consistency
  testing.
