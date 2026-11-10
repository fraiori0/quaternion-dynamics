# Attitude estimation in the same coordinates: an error-state filter that coincides with the right-invariant EKF

## Summary

The companion document *Linear dynamical systems for rotations* (`linear_systems_on_SO3.md`) introduces coordinates in which tracking a moving orientation becomes an exactly linear problem. The key ingredients are a world-frame error rotation, its logarithm, and the exact kinematic relation between the rate of change of that logarithm and the two angular velocities involved.

This document applies the same ingredients to a different problem: estimating orientation and gyro bias from an IMU (gyroscope plus direction sensors such as an accelerometer measuring gravity and a magnetometer). The result is:

- **Exact error dynamics.** The estimation error obeys $\dot\xi = -J_r^{-1}(\xi)\,\hat R\,(\delta b + n)$, with no approximation. The true angular velocity does not appear.
- **A single linearization.** The resulting Kalman filter linearizes in exactly one place in the propagation (setting $J_r^{-1}(\xi)\approx I$), and the term being neglected is written explicitly.
- **Equivalence with the RI-EKF.** The filter is algebraically identical to the right-invariant extended Kalman filter (RI-EKF) from the invariant-filtering literature. In simulation the two produce the same estimates to $10^{-12}$ rad.
- **Relation to the MEKF.** Compared with the classical multiplicative EKF (MEKF), the linearized models are the same up to a change of coordinates. The two filters differ only in how the covariance is carried across the step where a correction is applied to the estimate. This explains the RI-EKF's better behavior under large errors, which we reproduce numerically.

None of the resulting filter is new. The contribution is a short, elementary derivation of a known filter from the same coordinates used for tracking, with every approximation made visible.

---

## 1. Notation

The notation follows the companion document; the essentials are repeated here so that this document can be read on its own.

**Quaternions and rotations.**

- **Representation.** Orientations are unit quaternions, Hamilton convention, $q=(q_s,q_v)$. A rotation by angle $\theta$ about unit axis $n$ is $q=(\cos\frac\theta2, n\sin\frac\theta2)$, and $q$ and $-q$ are the same rotation.
- **Product and conjugate.** $\otimes$ is the quaternion product and $q^*$ the conjugate.
- **Rotation matrix.** $R(q)$ is the rotation matrix of $q$; it maps body-frame vectors to world-frame vectors.

**Exponential and logarithm.** For $r\in\mathbb R^3$ with $\theta=\|r\|$:

$$
\mathrm{Exp}(r)=\big(\cos\tfrac\theta2,\ \tfrac{\sin(\theta/2)}{\theta}r\big).
$$

$\mathrm{Log}(p)=\theta n$ is its inverse. It is computed after choosing the sign with $p_s\ge0$, so that $\theta\in[0,\pi]$. As a matrix, $\mathrm{Exp}(r)$ is the rotation by $\|r\|$ about $r/\|r\|$.

**Cross-product matrix.** $[v]_\times u = v\times u$.

**Jacobians of SO(3).** With $\theta=\|r\|$ and $S=[r]_\times$:

$$
J_r(r) = I - \frac{1-\cos\theta}{\theta^2}S + \frac{\theta-\sin\theta}{\theta^3}S^2,\qquad
J_l(r) = J_r(-r) = \mathrm{Exp}(r)\,J_r(r).
$$

If a rotation evolves as $\mathrm{Exp}(r(t))$, its world-frame angular velocity is $J_l(r)\dot r$ and its body-frame angular velocity is $J_r(r)\dot r$. Both Jacobians tend to $I$ as $r\to0$, and $J_r^{-1}(r) = I + \frac12[r]_\times + O(\|r\|^2)$.

**The kinematic relation used below.** This is relation (V) of the companion document, written for world-frame angular velocities. Let a "target" rotation $q_d$ have world angular velocity $\omega_d$ and a "tracker" rotation $q$ have world angular velocity $\omega_w$. Define $q_e=q_d\otimes q^*$, $R_e=R(q_e)$ and $r=\mathrm{Log}(q_e)$. Then

$$
\dot r = J_l^{-1}(r)\,\big(\omega_d - R_e\,\omega_w\big) = J_r^{-1}(r)\,\big(R_e^\top\omega_d - \omega_w\big). \tag{V}
$$

The first form holds because $\omega_d - R_e\omega_w$ is the world-frame angular velocity of $q_e$. The second follows from $J_l^{-1}=J_r^{-1}R_e^\top$.

---

## 2. The estimation problem

**State.**
- the true orientation $q$, with $R = R(q)$;
- the gyro bias $b\in\mathbb R^3$, in the body frame.

**Kinematics.** With $\omega$ the true body-frame angular velocity:

$$
\dot q = \tfrac12\, q\otimes\omega .
$$

**Gyroscope.** It is sampled with period $\Delta t$:

$$
\omega_{m,k} = \omega_k + b_k + n_k,\qquad n_k\sim\mathcal N(0,\sigma_g^2 I).
$$

**Bias.** It is modeled as a random walk:

$$
b_{k+1} = b_k + w_k,\qquad w_k\sim\mathcal N(0,\sigma_b^2\Delta t\, I).
$$

**Direction sensors.** A sensor measures, in the body frame, a known world-frame unit vector $u_i$:

$$
y_i = R^\top u_i + v_i,\qquad v_i\sim\mathcal N(0,\sigma_v^2 I).
$$

Typical choices are $u_1$ = "up" for an accelerometer, valid when linear acceleration is negligible compared to gravity, and $u_2$ = the local magnetic field direction for a magnetometer. Rotations about a single $u_i$ are unobservable from that sensor alone, so the full orientation needs at least two non-parallel directions.

**Estimator.** The filter maintains
- an estimate $\hat q$ of the orientation, with $\hat R = R(\hat q)$;
- an estimate $\hat b$ of the bias;
- a $6\times6$ covariance $P$ of the error state defined in Section 3.

Between measurements the estimate is propagated with the bias-corrected gyro rate:

$$
\hat q_{k+1} = \hat q_k\otimes \mathrm{Exp}\big((\omega_{m,k}-\hat b_k)\,\Delta t\big),\qquad \hat b_{k+1}=\hat b_k .
$$

---

## 3. Error coordinates

The **attitude error** is the world-frame rotation that takes the estimate to the truth, expressed by its logarithm:

$$
\xi = \mathrm{Log}\big(q\otimes\hat q^*\big)\in\mathbb R^3,\qquad\text{equivalently}\qquad q = \mathrm{Exp}(\xi)\otimes\hat q .
$$

The **bias error** is $\delta b = b - \hat b$.

The error state is $x=(\xi,\delta b)\in\mathbb R^6$. Note that $\|\xi\|$ is the angle between true and estimated orientation.

**Correspondence with the tracking document.**
- the truth $q$ plays the role of the target $q_d$;
- the estimate $\hat q$ plays the role of the tracker;
- $\xi$ plays the role of $r$.

The difference is that $\xi$ is unknown: the filter only has a mean and a covariance for it.

---

## 4. Exact error dynamics

Apply (V) with the following substitutions:
- target = truth, whose world angular velocity is $R\,\omega$;
- tracker = estimate, whose world angular velocity is $\hat R\,(\omega_m - \hat b)$;
- error rotation $R_e = R(q\otimes\hat q^*) = R\hat R^\top = \mathrm{Exp}(\xi)$.

Define the rate error, the difference between the rate used by the estimate and the true rate:

$$
\varepsilon = (\omega_m-\hat b)-\omega = \delta b + n .
$$

Then the world-frame angular velocity of the error rotation is

$$
R\,\omega - R_e\,\hat R\,(\omega_m-\hat b) = R\,\omega - R\hat R^\top\hat R\,(\omega+\varepsilon) = -R\,\varepsilon .
$$

Using the first form of (V), $R = R_e\hat R$, and $J_l^{-1}(\xi)R_e = J_r^{-1}(\xi)$:

$$
\boxed{\ \dot\xi = -J_r^{-1}(\xi)\,\hat R\,\big(\delta b + n\big),\qquad \dot{\delta b} = \text{bias random-walk noise}\ } \tag{E}
$$

Equation (E) involves no approximation. Three properties follow directly:

1. **No dependence on the motion.** The true angular velocity $\omega$ has cancelled. The error dynamics depend only on the known estimate $\hat R$, on the bias error, and on noise.
2. **Perfect rates preserve any error.** With a perfectly calibrated, noise-free gyro ($\varepsilon=0$), $\dot\xi = 0$ exactly: integrating perfect rates keeps the initial error constant, however large it is.
3. **One nonlinear factor.** The only nonlinearity is the factor $J_r^{-1}(\xi)$ multiplying the rate error.

---

## 5. The filter

### 5.1 Linearized propagation

The filter linearizes (E) around its mean error. After every update the mean error is reset to zero (Section 5.4), so the linearization point is $\xi = 0$, where $J_r^{-1}=I$:

$$
\dot\xi \approx -\hat R\,\delta b - \hat R\, n .
$$

The neglected part of (E) is

$$
-\big(J_r^{-1}(\xi)-I\big)\hat R\,\varepsilon = -\tfrac12\,\xi\times\big(\hat R\,\varepsilon\big) + O\big(\|\xi\|^2\|\varepsilon\|\big),
$$

a product of the attitude error with the bias error or the noise. This is the only approximation in the propagation.

Discretized over one gyro sample, with $\hat R_k$ evaluated at the start of the step:

$$
x_{k+1} = F_k\, x_k + G_k\, \nu_k,\qquad
F_k = \begin{pmatrix} I & -\hat R_k\Delta t\\ 0 & I\end{pmatrix},\quad
G_k = \begin{pmatrix} -\hat R_k\Delta t & 0\\ 0 & I\end{pmatrix},\quad
\nu_k = \begin{pmatrix} n_k\\ w_k\end{pmatrix},
$$

$$
P \leftarrow F_k P F_k^\top + G_k\, Q\, G_k^\top,\qquad Q = \mathrm{diag}\big(\sigma_g^2 I,\ \sigma_b^2\Delta t\, I\big).
$$

Because $\hat R_k$ is a rotation, the attitude part of the process noise is simply $\sigma_g^2\Delta t^2 I$.

### 5.2 Linearized measurement

With $q=\mathrm{Exp}(\xi)\otimes\hat q$, the true rotation matrix is $R = \mathrm{Exp}(\xi)\hat R$, so

$$
y_i = \hat R^\top\,\mathrm{Exp}(-\xi)\,u_i + v_i .
$$

Expanding $\mathrm{Exp}(-\xi)u_i = u_i - \xi\times u_i + O(\|\xi\|^2) = u_i + [u_i]_\times\xi + O(\|\xi\|^2)$ gives the innovation and Jacobian

$$
z_i = y_i - \hat R^\top u_i \approx \hat R^\top [u_i]_\times\, \xi + v_i,\qquad
H_i = \begin{pmatrix}\hat R^\top [u_i]_\times & 0\end{pmatrix}.
$$

The bias does not enter the measurement directly. The null space of $[u_i]_\times$ is the direction $u_i$, matching the unobservability of rotations about the measured direction. Unlike the propagation, the measurement cannot be made linear in any coordinates, since $y_i$ is a bounded, periodic function of $\xi$. This linearization is the second and last approximation of the filter.

### 5.3 Update

Stack all measurements available at a time step into $z$, $H$ and noise covariance $\Sigma_v$ (here $\sigma_v^2 I$). The standard Kalman update gives the error estimate and the new covariance:

$$
S = H P H^\top + \Sigma_v,\qquad K = P H^\top S^{-1},\qquad
\begin{pmatrix}\hat\xi^+\\ \delta\hat b^+\end{pmatrix} = K z,\qquad P\leftarrow (I-KH)P .
$$

### 5.4 Injection (moving the correction into the estimate)

The estimated error is applied to the nominal estimate, consistently with the error definition $q = \mathrm{Exp}(\xi)\otimes\hat q$:

$$
\hat q \leftarrow \mathrm{Exp}(\hat\xi^+)\otimes\hat q,\qquad \hat b\leftarrow\hat b+\delta\hat b^+ .
$$

After injection the mean error is zero again, which is the linearization point assumed in Sections 5.1 and 5.2.

**Optional covariance correction.** Injection changes the reference of the error. The new error is

$$
\xi_{\text{new}} = \mathrm{Log}\big(\mathrm{Exp}(\xi)\,\mathrm{Exp}(-\hat\xi^+)\big),
$$

whose first-order dependence on $\xi$ around $\hat\xi^+$ is $J_l(\hat\xi^+)$. A first-order-consistent filter therefore also applies

$$
P\leftarrow T P T^\top,\qquad T = \mathrm{diag}\big(J_l(\hat\xi^+),\ I\big).
$$

This is the covariance correction step studied for the MEKF by Mueller, Hehn and D'Andrea. In the tracking document the same operation appears as re-expressing the chart state when the goal changes. The filter compared in Section 8 omits this step, as the RI-EKF is usually implemented; Section 8 also reports its effect.

### 5.5 Algorithm

Per gyro sample:

1. Propagate $P$ with $F_k$, $G_k$ evaluated at the current $\hat q$.
2. Propagate $\hat q\leftarrow\hat q\otimes\mathrm{Exp}((\omega_m-\hat b)\Delta t)$.

Per measurement:

1. Compute $z_i=y_i-\hat R^\top u_i$ and $H_i$.
2. Run the Kalman update.
3. Inject $\hat q\leftarrow\mathrm{Exp}(\hat\xi^+)\otimes\hat q$ and $\hat b\leftarrow\hat b+\delta\hat b^+$.
4. Optionally, apply the covariance correction.

---

## 6. Equivalence with the right-invariant EKF

In the invariant-filtering framework of Barrau and Bonnabel, the orientation is a Lie group element $\chi=R$. A measurement of the form $y=\chi^{-1}u+v$ (a known world vector seen in the body frame) is paired with the right-invariant error. This error is a world-frame rotation between truth and estimate, $\eta = R\hat R^\top$ or its inverse depending on the author's convention; using the inverse only flips the sign of $\xi$ and leaves the filter unchanged. The RI-EKF linearizes $\eta = \mathrm{Exp}(\xi)$, propagates the same error-state model as Section 5.1, and injects corrections as $\hat R\leftarrow\mathrm{Exp}(\hat\xi^+)\hat R$.

The only visible difference is the measurement step. The RI-EKF rotates the innovation into the world frame:

$$
z_i^{\text{RI}} = \hat R\,y_i - u_i \approx [u_i]_\times\,\xi + \hat R\, v_i,\qquad H_i^{\text{RI}} = \begin{pmatrix}[u_i]_\times & 0\end{pmatrix},\qquad \Sigma^{\text{RI}}_{v,i} = \hat R\,\Sigma_{v,i}\hat R^\top .
$$

Its Jacobian is constant, which is a hallmark of the invariant approach.

**Claim.** The two updates are identical.

**Proof.** Let $M=\mathrm{diag}(\hat R,\dots,\hat R)$ be the block-diagonal orthogonal matrix that rotates every measurement block. Then

$$
z^{\text{RI}} = M z,\qquad H^{\text{RI}} = M H,\qquad \Sigma_v^{\text{RI}} = M\Sigma_v M^\top ,
$$

so $S^{\text{RI}} = M S M^\top$. Substituting:

- the gain becomes $K^{\text{RI}} = P H^\top M^\top (M S M^\top)^{-1} = K M^{-1}$;
- the correction becomes $K^{\text{RI}}z^{\text{RI}} = K z$;
- the covariance update becomes $I-K^{\text{RI}}H^{\text{RI}} = I-KH$.

The propagation and injection steps are the same in both filters, so the two filters coincide step by step. The argument holds for any measurement noise covariance, provided the RI-EKF rotates it as above. For isotropic noise, $\hat R\,\sigma_v^2 I\,\hat R^\top = \sigma_v^2 I$, so no rotation is needed.

**Numerically**, over 240 simulated runs including initial errors of arbitrary size (Section 8), the final estimates of the two implementations differ by at most $7\times10^{-13}$ rad.

**What the derivation adds.** The filter is the same, but the route to it is different and elementary:

- **No Lie-group machinery.** It needs only the exact kinematic relation (V) and one Taylor expansion.
- **The neglected term is explicit.** It is $-\frac12\,\xi\times\hat R(\delta b+n)$.
- **It shows where exactness is lost.** Without bias, the deterministic part of (E) is exactly zero, an instance of the "log-linear" error property that is central to invariant filtering. With bias, the error dynamics contain the product $J_r^{-1}(\xi)\hat R\,\delta b$, which is not linear in $(\xi,\delta b)$. This is consistent with the bias case being treated in the literature as an approximately (not exactly) invariant problem.

---

## 7. Relation to the multiplicative EKF

The classical MEKF uses the body-frame error $\delta = \mathrm{Log}(\hat q^*\otimes q)$, i.e. $q = \hat q\otimes\mathrm{Exp}(\delta)$. This is the left-invariant error. Since $q = \mathrm{Exp}(\xi)\otimes\hat q$, the two errors are related exactly by

$$
\delta = \hat R^\top\xi .
$$

Under this change of coordinates the two filters' models agree:

- **Propagation.** Differentiating $\delta=\hat R^\top\xi$ with (E) gives $\dot\delta = -[\hat\omega]_\times\delta - J_r^{-1}(\delta)(\delta b+n)$, where $\hat\omega=\omega_m-\hat b$ and $\hat R^\top J_r^{-1}(\xi)\hat R=J_r^{-1}(\delta)$. Its linearization, $\dot\delta\approx-[\hat\omega]_\times\delta - \delta b - n$, is the standard MEKF model; it neglects the same term.
- **Measurement.** $\hat R^\top[u]_\times\xi = [\hat R^\top u]_\times\delta$, which is the standard MEKF Jacobian.
- **Injection of the mean.** $\hat q\otimes\mathrm{Exp}(\hat\delta^+) = \mathrm{Exp}(\hat\xi^+)\otimes\hat q$ exactly.

The linearized propagation and measurement models are therefore the same model, written in two coordinate systems related by the rotation $\hat R$ (up to an $O(\Delta t^2)$ difference in discretization).

**Where they differ: carrying the covariance across injection.**

- **World-frame filter.** The covariance of $\xi$ is expressed in the world frame, which does not move when the estimate is corrected, so the filter keeps $P$ unchanged.
- **MEKF.** The covariance of $\delta$ is expressed in the frame of the estimate. When the estimate jumps by $\hat\delta^+$, keeping $P$ unchanged amounts to rotating it relative to the world-frame filter. Translated into body coordinates, the world-frame filter's choice corresponds to $P_\delta\leftarrow C P_\delta C^\top$ on the attitude rows and columns, with $C=\mathrm{Exp}(-\hat\delta^+)$; the MEKF applies no such rotation.
- **First-order-consistent reset.** The reset of Section 5.4 lies between the two choices. In world coordinates it is $J_l(\hat\xi^+)\approx I+\frac12[\hat\xi^+]_\times$; in body coordinates it is $J_r(\hat\delta^+)\approx I-\frac12[\hat\delta^+]_\times$. These two corrections are equivalent to each other.

All three variants coincide when corrections are small, which is why all filters behave identically in normal operation. They differ only when individual corrections are large, i.e. during convergence from a large initial error. Section 8 confirms numerically both the equivalences and the differences.

---

## 8. Numerical comparison

### 8.1 Setup

- **Truth.** Random smooth body rates, each component a sum of two sinusoids with amplitudes in $[0.2,1.2]$ rad/s and frequencies in $[0.1,1.5]$ rad/s; these are scaled by 4 in the fast-motion scenario.
- **Sensors.**
  - gyro at 100 Hz, $\sigma_g=0.01$ rad/s per sample;
  - bias random walk $\sigma_b=10^{-3}$ rad/s/$\sqrt{\text{s}}$;
  - two direction sensors at 10 Hz (gravity "up" and a magnetic field direction inclined by $53^\circ$), $\sigma_v=0.03$;
  - initial bias drawn with $\sigma=0.02$ rad/s (0.05 in the fast-motion scenario).
- **Runs.** 15 s each, 60 Monte Carlo runs per scenario. All filters receive identical sensor streams and use the true noise parameters.
- **Initialization.** The initial estimate error is drawn from the filters' initial covariance, except in the unknown-initialization scenario.

**Filters.**

| Name | Description |
|---|---|
| AQ | Additive-quaternion EKF: the 4-D quaternion is a filter state, renormalized after each update. |
| MEKF | Body-frame error, no covariance correction at injection. |
| RI-EKF | Right-invariant EKF in its standard form (Section 6). |
| This filter | Sections 5.1–5.4, without the optional covariance correction. |

**Scenarios.**

| Scenario | Initial attitude error |
|---|---|
| nominal | $\sigma=5^\circ$ per axis |
| large_init | $\sigma=60^\circ$ per axis |
| unknown_init | uniformly random initial orientation (filter initialized with $\sigma=\pi/\sqrt3$ per axis) |
| fast_motion | rates ×4, $\sigma=20^\circ$ per axis |

**Metrics.**

- **RMSE:** attitude error in the settled window 8–15 s.
- **Converged:** the fraction of runs whose final attitude error is below $2^\circ$.
- **NEES:** the median normalized estimation error squared of the attitude, $e^\top P_{\text{att}}^{-1}e$, over 8–15 s. It checks whether the covariance matches the actual error; the ideal value is 3, and larger values mean overconfidence.

### 8.2 Results

The RI-EKF and this filter are numerically identical in all runs (maximum difference $7\times10^{-13}$ rad), so they share one column.

| Scenario | Metric | AQ | MEKF | RI-EKF = this filter |
|---|---|---|---|---|
| nominal | RMSE | 0.66° | 0.66° | 0.66° |
| | converged | 100% | 100% | 100% |
| | NEES | 2.6 | 2.6 | 2.6 |
| large_init | RMSE | 22.3° | 5.5° | 3.0° |
| | converged | 43% | 68% | 95% |
| | NEES | 142 | 13 | 4.4 |
| unknown_init | RMSE | 55.5° | 14.5° | 5.7° |
| | converged | 23% | 53% | 78% |
| | NEES | 793 | 45 | 9.6 |
| fast_motion | RMSE | 0.81° | 0.71° | 0.68° |
| | converged | 100% | 100% | 100% |
| | NEES | 3.1 | 2.7 | 2.5 |

**Injection variants (large_init).** These runs isolate the mechanism described in Section 7. The first block uses the first 40 runs and the second block the first 30.

| Variant | RMSE | Converged | NEES |
|---|---|---|---|
| MEKF (no covariance change at injection) | 5.42° | 65% | 17.2 |
| MEKF + rotation $C=\mathrm{Exp}(-\hat\delta^+)$ at injection | 3.45° | 95% | 5.05 |
| This filter (no covariance change) | 3.51° | 95% | 5.07 |
| MEKF + first-order reset $J_r(\hat\delta^+)$ | 4.12° | 90% | 9.98 |
| This filter + first-order reset $J_l(\hat\xi^+)$ | 4.20° | 90% | 10.03 |

Within each matched pair, final estimates differ by a median below $0.01^\circ$, consistent with the $O(\Delta t^2)$ discretization difference.

### 8.3 Interpretation

- **Normal operation.** With small errors, all four filters are equivalent.
- **Large errors.** The world-frame error, i.e. the RI-EKF, converges more often and remains far better calibrated than the MEKF and the additive-quaternion EKF. This is the known advantage of the invariant EKF.
- **The mechanism.** The difference between the MEKF and the world-frame filter is entirely the treatment of the covariance at injection. Adding the rotation $C$ to the MEKF reproduces the world-frame filter, and the two first-order resets are equivalent to each other, as predicted in Section 7.
- **An open point.** The first-order-consistent reset sits between the two simple choices in performance. Under large errors, first-order consistency does not translate into faster convergence. Explaining this would require analysis beyond first order and is not attempted here.

---

## 9. Scope and limitations

- **Simulation only.** The comparison uses one sensor configuration, noise parameters equal to the true ones, and 60 runs per scenario. Validation on recorded IMU data with ground-truth orientation would strengthen it.
- **Simple bias treatment.** The bias is handled in the simplest way, identically in all error-state filters. Invariant-filtering variants that treat the bias differently were not compared.
- **Accelerometer model.** Using the accelerometer as a gravity sensor assumes negligible linear acceleration. Handling acceleration requires a larger state (velocity, possibly position), where the invariant framework is usually applied to extended groups.
- **No novelty in the filter.** The filter itself is the RI-EKF; only the derivation route is different.

---

## Appendix: reference implementation (Python)

Quaternions are `[w, x, y, z]`. Gyro rates and direction measurements are in the body frame; the reference directions `us` are world-frame unit vectors. `P` is the covariance of $(\xi,\delta b)$.

```python
import numpy as np

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def Exp(r):                                   # rotation vector -> unit quaternion
    th = np.linalg.norm(r)
    return np.concatenate([[np.cos(th/2)], 0.5*np.sinc(th/(2*np.pi))*r])

def Rot(q):                                   # unit quaternion -> rotation matrix
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def predict(qh, bh, P, w_m, dt, sigma_g, sigma_b):
    """one gyro sample. P is the 6x6 covariance of (xi, db)."""
    Rh = Rot(qh)
    F = np.eye(6);        F[:3, 3:] = -Rh*dt
    G = np.zeros((6, 6)); G[:3, :3] = -Rh*dt; G[3:, 3:] = np.eye(3)
    Q = np.diag([sigma_g**2]*3 + [sigma_b**2*dt]*3)
    P = F @ P @ F.T + G @ Q @ G.T
    qh = qmul(qh, Exp((w_m - bh)*dt)); qh /= np.linalg.norm(qh)
    return qh, bh, P

def update(qh, bh, P, ys, us, sigma_v):
    """direction measurements ys[i] (body frame) of known world unit vectors us[i]."""
    Rh = Rot(qh); m = len(ys)
    z = np.concatenate([y - Rh.T @ u for y, u in zip(ys, us)])
    H = np.zeros((3*m, 6))
    for i, u in enumerate(us): H[3*i:3*i+3, :3] = Rh.T @ hat(u)
    S = H @ P @ H.T + sigma_v**2*np.eye(3*m)
    K = np.linalg.solve(S, H @ P).T
    dx = K @ z
    P = (np.eye(6) - K @ H) @ P; P = 0.5*(P + P.T)
    qh = qmul(Exp(dx[:3]), qh); qh /= np.linalg.norm(qh)   # injection: world-frame correction
    bh = bh + dx[3:]
    return qh, bh, P
```
