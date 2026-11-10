# Linear dynamical systems for rotations: an exact change of coordinates for tracking on SO(3)

## Summary

Many trajectory generators and controllers are linear systems written in terms of a tracking error and its time derivatives: a spring-mass-damper, an LQR tracker, an online minimum-jerk generator with time-to-go gains, and so on. For positions they are straightforward to use, because position errors live in $\mathbb R^3$ and add and subtract linearly. For orientations they are not, because rotations do not commute.

This document defines a change of coordinates between the rotational state of a tracker relative to a (possibly moving) target and a vector state in $\mathbb R^{3n}$. In these coordinates the rotational tracking problem *is* a linear tracking problem in $\mathbb R^3$. As a result:

- any linear tracking system (time-invariant or time-varying, of any order) can be run on rotations without modification;
- the error vector follows exactly the trajectory the same linear system would produce for a position error;
- the angle between tracker and target equals the norm of that linear error at all times;
- this holds for arbitrary target motion, including targets that tumble about a changing axis.

When the target rotates about a fixed axis and the tracker moves about the same axis, the construction reduces to a very simple form without Jacobians (Section 7).

---

## 1. Setting and notation

![Tracker, target and error rotation](so3_tracking_figure.png)

*Figure: the tracker (solid) and the target (dashed) are two orientations in the world frame. The error rotation $q_e$ takes the tracker to the target; it is a rotation by the angle $\theta$ about the unit axis $n$, both expressed in the world frame. The error vector is $r=\theta n$.*

**Quaternions.** Orientations are unit quaternions $q = (q_s, q_v)$ with scalar part $q_s\in\mathbb R$ and vector part $q_v\in\mathbb R^3$, Hamilton convention. A rotation by angle $\theta$ about unit axis $n$ is $q = (\cos\frac\theta2,\ n\sin\frac\theta2)$. $\otimes$ is the quaternion product, $q^*=(q_s,-q_v)$ the conjugate (the inverse, for unit quaternions). $q$ and $-q$ represent the same rotation. $R(q)\in SO(3)$ is the rotation matrix of $q$.

**Angular velocities** are expressed in the **world frame** throughout. A vector $\omega\in\mathbb R^3$ in a quaternion product means the pure quaternion $(0,\omega)$. The kinematics are

$$
\dot q = \tfrac12\,\omega\otimes q .
$$

**Tracker and target.**

| Symbol | Meaning |
|---|---|
| $q,\ \omega,\ \dot\omega,\ \ddot\omega,\dots$ | orientation of the tracker and its angular velocity, acceleration, jerk, ... |
| $q_d,\ \omega_d,\ \dot\omega_d,\dots$ | the same quantities for the target |

The tracker is the rotational system we want to generate or control. The target can move in any way; its derivatives up to the order required by the chosen linear system (Section 4) are assumed known.

**Exponential and logarithm.** For a rotation vector $r\in\mathbb R^3$ with $\theta=\|r\|$:

$$
\mathrm{Exp}(r) = \Big(\cos\tfrac\theta2,\ \tfrac{\sin(\theta/2)}{\theta}\, r\Big),
$$

and for a unit quaternion $p$, first choose the sign with $p_s\ge 0$ (shortest rotation), then

$$
\theta = 2\,\mathrm{atan2}\big(\|p_v\|,\ p_s\big)\in[0,\pi],\qquad
\mathrm{Log}(p) = \frac{\theta}{\sin(\theta/2)}\, p_v = \theta\, n .
$$

The ratios $\sin(\theta/2)/\theta$ and $\theta/\sin(\theta/2)$ are evaluated with a sinc function so that they are well defined at $\theta=0$ (limits $\tfrac12$ and $2$).

**Cross-product matrix.** $[v]_\times$ is the $3\times3$ matrix with $[v]_\times u = v\times u$.

---

## 2. The linear system to be transferred

In position space, a tracker at position $p$ follows a target at $p_d$. With the error $e = p_d - p \in \mathbb R^3$, a linear tracking system of order $n$ is an ODE on the error:

$$
e^{(n)} = \sum_{k=0}^{n-1} K_k(t)\, e^{(k)}, \qquad K_k(t)\in\mathbb R^{3\times 3}. \tag{L}
$$

The gains may be constant or time-varying, scalar or full matrices. The linear state is $x = (e,\dot e,\dots,e^{(n-1)})\in\mathbb R^{3n}$. Equation (L) describes a tracker that knows the target's derivatives up to order $n$ (full feedforward); the tracker itself is recovered from $p = p_d - e$. Additional internal states of the linear system (integral action, filters) can be added; they are carried along unchanged.

Examples:

- **Spring-mass-damper:** $\ddot e = -K e - D\dot e$.
- **LQR tracking:** $\ddot e = -K_p e - K_d \dot e$, where $(K_p, K_d)$ is the LQR gain for the double integrator $\ddot e = v$ with cost $\int \big(x^\top Q_x\, x + v^\top Q_v\, v\big)\,dt$.
- **Online minimum-jerk** (Hoff–Arbib): with time-to-go $\tau = T - t$,
  $\dddot e = -\big(\tfrac{60}{\tau^3}e + \tfrac{36}{\tau^2}\dot e + \tfrac{9}{\tau}\ddot e\big)$.
  From any initial state, $e$, $\dot e$, $\ddot e$ reach zero at $t=T$ along the minimum-jerk profile.

The goal is to run exactly these equations for orientations.

---

## 3. The change of coordinates

### 3.1 Error rotation and error vector

The **error rotation** takes the tracker to the target in the world frame:

$$
q_e = q_d \otimes q^* .
$$

The **error vector** is its logarithm:

$$
r = \mathrm{Log}(q_e) = \theta\, n .
$$

$\theta = \|r\|\in[0,\pi]$ is the angle of the rotation from tracker to target (the geodesic distance on SO(3)), and $n$ is its axis in the world frame. $r$ plays the role of the position error $e$. We write $R_e = R(q_e) = \mathrm{Exp}(r)$ as a matrix.

### 3.2 Chart state

The **chart state** collects $r$ and its true time derivatives:

$$
\xi = \big(r,\ \dot r,\ \ddot r,\ \dots,\ r^{(n-1)}\big)\in\mathbb R^{3n}.
$$

Note that $\dot r$ is the time derivative of the vector $r$, **not** an angular velocity and in general not equal to $\omega_d-\omega$. Section 3.4 gives the exact relations.

### 3.3 Jacobians of SO(3)

If a rotation evolves as $\mathrm{Exp}(r(t))$, its angular velocity is not $\dot r$; the two are related by the **left and right Jacobians** of SO(3):

$$
\omega_{\text{world}} = J_l(r)\,\dot r, \qquad \omega_{\text{body}} = J_r(r)\,\dot r .
$$

With $\theta=\|r\|$ and $S=[r]_\times$:

$$
J_r(r) = I - a(\theta)\, S + b(\theta)\, S^2,\qquad a(\theta) = \frac{1-\cos\theta}{\theta^2},\quad b(\theta) = \frac{\theta-\sin\theta}{\theta^3},
$$

$$
J_l(r) = J_r(-r) = R_e\, J_r(r),
$$

$$
J_r^{-1}(r) = I + \tfrac12 S + \Big(\frac1{\theta^2} - \frac{1+\cos\theta}{2\theta\sin\theta}\Big) S^2 .
$$

Properties used below:

- $J_r(r)\,r = J_l(r)\,r = r$: along the axis of rotation the Jacobians act as the identity.
- Near $\theta = 0$: $a\to\frac12$, $b\to\frac16$, so $J_r\to I$.
- $J_r$ is invertible for $\theta<2\pi$, hence everywhere in the domain $\theta\le\pi$ used here.

The time derivative of $J_r$ along a trajectory $r(t)$ is

$$
\dot J_r = -\dot a\, S - a\,\dot S + \dot b\, S^2 + b\,(\dot S S + S\dot S),\qquad \dot S = [\dot r]_\times,
$$

with $\dot a = a'(\theta)\,\dot\theta$, $\dot b = b'(\theta)\,\dot\theta$, $\dot\theta = r^\top\dot r/\theta$, and

$$
a'(\theta) = \frac{\sin\theta}{\theta^2} - \frac{2(1-\cos\theta)}{\theta^3},\qquad
b'(\theta) = \frac{1-\cos\theta}{\theta^3} - \frac{3(\theta-\sin\theta)}{\theta^4}.
$$

Near $\theta=0$ use $a'(\theta)/\theta\to -\frac1{12}$ and $b'(\theta)/\theta \to -\frac1{60}$ (so that $\dot a = (a'/\theta)\, r^\top \dot r$ needs no division by $\theta$).

### 3.4 Kinematics of the error

Differentiating $q_e = q_d\otimes q^*$ with $\dot q = \frac12\omega\otimes q$ and $\dot q_d = \frac12\omega_d\otimes q_d$ gives

$$
\dot q_e = \tfrac12\,(\omega_d - R_e\,\omega)\otimes q_e ,
$$

so the world-frame angular velocity of the error rotation is $\omega_d - R_e\omega$. Applying $J_l^{-1}$ and using $J_l^{-1}=J_r^{-1}R_e^\top$:

$$
\boxed{\ \dot r = J_r^{-1}(r)\,\big(R_e^\top \omega_d - \omega\big)\ } \tag{V}
$$

and, solved for the tracker's angular velocity,

$$
\boxed{\ \omega = R_e^\top\omega_d - J_r(r)\,\dot r\ } \tag{V'}
$$

Differentiating (V') once more, with $\frac{d}{dt}R_e^\top = -R_e^\top\,[J_l(r)\,\dot r]_\times$:

$$
\boxed{\ \dot\omega = -R_e^\top [J_l(r)\,\dot r]_\times\,\omega_d + R_e^\top\dot\omega_d - \dot J_r\,\dot r - J_r(r)\,\ddot r\ } \tag{A'}
$$

and, solved for $\ddot r$,

$$
\ddot r = J_r^{-1}(r)\Big(-R_e^\top [J_l(r)\,\dot r]_\times\,\omega_d + R_e^\top\dot\omega_d - \dot J_r\,\dot r - \dot\omega\Big). \tag{A}
$$

**Higher orders.** Differentiating (V') $k$ times always gives

$$
\omega^{(k)} = -J_r(r)\, r^{(k+1)} + G_k\big(r,\dots,r^{(k)},\ \omega_d,\dots,\omega_d^{(k)}\big),
$$

where $G_k$ collects terms with lower derivatives of $r$, derivatives of $J_r$ and $R_e$ up to order $k$, and target derivatives. Because the coefficient of the highest derivative is always the invertible $J_r(r)$, each relation can be solved for $r^{(k+1)}$. Orders up to $\dot\omega$ (explicit above) cover second- and third-order systems such as LQR and minimum-jerk in the generator form of Section 5.

### 3.5 The coordinate map

Equations (V)–(A) and their higher-order versions define, for a given target state, a map

$$
\Phi:\ \big(q,\ \omega,\ \dot\omega,\ \dots,\ \omega^{(n-2)}\big)\ \longmapsto\ \xi = \big(r,\ \dot r,\ \ddot r,\ \dots,\ r^{(n-1)}\big),
$$

with explicit inverse

$$
\Phi^{-1}:\quad q = \mathrm{Exp}(-r)\otimes q_d,\quad \omega \text{ from (V')},\quad \dot\omega \text{ from (A')},\ \dots
$$

On the domain $\theta<\pi$, $\mathrm{Log}$ is smooth and every $J_r$ is invertible, so $\Phi$ is a smooth, invertible change of coordinates between the rotational tracking state and $\mathbb R^{3n}$. It depends on the target state, so it moves with the target.

---

## 4. Main result

**Statement.** Let a linear tracking system (L) of order $n$ be given, a target trajectory $q_d(t)$ with known derivatives, and an initial tracker state with $\theta(0)<\pi$. Define the rotational tracker by:

1. $\xi(0) = \Phi\big(\text{tracker state at } t=0\big)$;
2. $\xi(t)$ evolves by (L), i.e. $r^{(n)} = \sum_k K_k(t)\, r^{(k)}$;
3. the tracker state at time $t$ is $\Phi^{-1}\big(\xi(t)\big)$, using the target state at time $t$.

Then, as long as $\|r(t)\|<\pi$:

1. **It is a genuine rotational motion.** $q(t)$ is a unit-quaternion trajectory whose angular velocity is exactly $\omega(t)$, whose angular acceleration is exactly $\dot\omega(t)$, and so on up to the order produced by $\Phi^{-1}$, because $\Phi^{-1}$ is built from the exact kinematic relations of Section 3.4.
2. **The error follows the linear system exactly.** $r(t) = e(t)$, where $e(t)$ is the solution of (L) with initial state $x(0)=\xi(0)$.
3. **Angle to target equals the linear error norm.** $\theta(t) = \|e(t)\|$ for all $t$.
4. **Convergence transfers.** If $x(t)\to 0$, then $q\to q_d$, $\omega\to\omega_d$, $\dot\omega\to\dot\omega_d$, ..., because at $\xi=0$ the maps reduce to $R_e=I$, $J_r=I$, $\dot J_r = 0$. If $x$ reaches zero at a finite time (minimum-jerk at $t=T$), the tracker reaches the target at that time with matched angular velocity and acceleration.

**Why it holds.** $\Phi$ is a change of coordinates; in the coordinates $\xi$ the rotational tracker is the linear system by construction, and $\Phi^{-1}$ maps its trajectory back without approximation. No assumption is made on the target motion beyond smoothness and knowledge of its derivatives.

**What the linear quantities mean on rotations.**

- $\|r\| = \theta$ is the true angle to the target. Costs or gains on $r$ act on the actual angular error; for example, a scalar weight $c$ gives $r^\top (cI)\, r = c\,\theta^2$.
- $\dot r$, $\ddot r$ are derivatives of the error vector. They coincide with $\omega_d-\omega$ and $\dot\omega_d-\dot\omega$ when $r=0$, and along the error axis (Section 7), but not in general. A linear design therefore shapes the evolution of the error vector $r(t)$; for LQR, optimality is with respect to the cost written in $(r,\dot r,\ddot r)$.
- Matrix gains act on the components of $r$ in the world frame.

**Domain.** The construction requires $\theta<\pi$. The linear design must keep $\|e(t)\|<\pi$ (no overshoot through $\pi$). This is a topological limit of SO(3), not of the construction: no continuous coordinate system covers all rotations around a target.

**Required target information.** Outputting $(q,\omega,\dots,\omega^{(n-2)})$ needs the target state up to $\omega_d^{(n-2)}$. Commanding $\omega^{(n-1)}$ (controller form, Section 5.2) needs $\omega_d^{(n-1)}$. This is the same information a position tracker with full feedforward uses. If the linear model contains an exogenous input $\nu(t)$, i.e. $e^{(n)} = \sum_k K_k e^{(k)} + \nu$, the construction holds unchanged with $\nu$ entering the equation for $r^{(n)}$; when $\nu$ represents unknown target motion, note that the target's angular acceleration enters (A) through $J_r^{-1}R_e^\top = J_l^{-1}$, which equals the identity along the error axis but not across it.

---

## 5. Implementation

### 5.1 Generator form (trajectory generation)

Used when the tracker is a virtual system producing a reference $(q,\omega,\dot\omega,\dots)$, e.g. for a robot's orientation controller.

At initialization, and again whenever the target is replaced (a new goal, or a jump in the target trajectory):

- compute $\xi = \Phi(\text{current tracker state})$ with respect to the new target. The tracker state $(q,\omega,\dots,\omega^{(n-2)})$ is continuous across the change; only $\xi$ is re-expressed.

At each time step:

1. Integrate (L) on $\xi\in\mathbb R^{3n}$ with any integrator (explicit Euler is sufficient).
2. Output $q = \mathrm{Exp}(-r)\otimes q_d$, $\omega$ from (V'), $\dot\omega$ from (A'), using the current target state.

No quaternion integration or renormalization is needed, and a third-order generator requires only $R_e$, $J_r$, $J_l$ and $\dot J_r$.

### 5.2 Controller form (driving a rotating system)

Used when the tracker is a system commanded through its highest derivative, e.g. a rotating body commanded in angular acceleration ($n=2$).

At each time step:

1. Measure $(q,\omega)$ and compute $r$, $\dot r$ from (V).
2. Evaluate the linear law: $\ddot r_{\text{cmd}} = K_0 r + K_1 \dot r$.
3. Command $\dot\omega$ from (A') with $\ddot r = \ddot r_{\text{cmd}}$.

For a physical rigid body with inertia $I_b$, apply the commanded acceleration through computed torque, $\tau_b = I_b\dot\omega_b + \omega_b\times I_b\omega_b$, with $\omega_b = R(q)^\top\omega$ and $\dot\omega_b = R(q)^\top\dot\omega$. Higher-order controller forms need the next relation of Section 3.4 (for $n=3$: $\ddot\omega$, which involves $\ddot J_r$).

### 5.3 Examples

- **Rotational spring-mass-damper:** $\ddot r = -K r - D\dot r$, with the target fixed at the rest orientation. The angle to the rest orientation equals, at all times, the displacement norm of the linear spring-mass-damper started from the same chart state.
- **LQR on rotations:** solve the standard LQR for $\ddot e = v$ (per axis, or with full weight matrices $Q_x$, $Q_v$), and use $\ddot r = -K_p r - K_d\dot r$. The rotational tracker attains exactly the linear optimal cost, measured in $(r,\dot r,\ddot r)$.
- **Online minimum-jerk orientation tracking:** run the Hoff–Arbib equation on $\xi=(r,\dot r,\ddot r)$ with $\tau = T-t$, in generator form. The tracker reaches a moving target at $t=T$ with $\omega=\omega_d$ and $\dot\omega=\dot\omega_d$, and the angle to the target follows the minimum-jerk profile of $\|e(t)\|$. Since the gains diverge as $\tau\to0$, stop integrating when $\tau$ falls below a small threshold and hold $\xi=0$; the tracker then moves with the target. The goal or the deadline $T$ can be changed at any time by re-initializing $\xi$ (Section 5.1).

---

## 6. Practical notes

- **Hemisphere.** Always take $q_e$ with non-negative scalar part before computing $\mathrm{Log}$, so that $\theta\le\pi$.
- **Small angles.** Use the series limits given above for $a$, $b$, $a'/\theta$, $b'/\theta$ and the coefficient of $S^2$ in $J_r^{-1}$ (limit $\frac1{12}$) when $\theta$ is below roughly $10^{-4}$.
- **Body-frame rates.** If angular velocities are available in the body frame, convert them before using the formulas: $\omega = R(q)\,\omega_b$ and $\dot\omega = R(q)\,\dot\omega_b$ (the second identity holds because $\omega\times\omega = 0$); likewise for the target with $R(q_d)$. Convert outputs back with $R(q)^\top$.
- **Discretization.** The linear system is integrated in $\mathbb R^{3n}$, so its discrete behavior is that of the chosen integrator applied to (L); the maps $\Phi$, $\Phi^{-1}$ are evaluated exactly at each step.

---

## 7. Special case: single-axis motion

Suppose that for a fixed unit axis $n$ (world frame) and a fixed orientation $\bar q$:

- the target rotates about $n$: $q_d(t) = \mathrm{Exp}\big(\sigma(t)\,n\big)\otimes\bar q$, so $\omega_d = \dot\sigma\, n$;
- the tracker starts on the same family: $q(0) = \mathrm{Exp}(\psi_0\, n)\otimes\bar q$, with $\omega(0),\dot\omega(0),\dots$ all parallel to $n$;
- the gains $K_k(t)$ preserve the direction $n$ (scalar gains, or matrices having $n$ as a common eigenvector).

Then the tracker stays on the family, $q(t) = \mathrm{Exp}(\psi(t)\,n)\otimes\bar q$. The error is $q_e = \mathrm{Exp}(s\,n)$ with the signed angle $s=\sigma-\psi$, so $r = s\,n$. All vectors involved are parallel to $n$, and therefore $R_e n = n$, $J_r n = J_l n = n$, $\dot J_r \dot r = 0$, and $[J_l\dot r]_\times\omega_d = 0$. The relations of Section 3.4 reduce to

$$
r = s\,n,\qquad \dot r = \omega_d - \omega,\qquad \ddot r = \dot\omega_d - \dot\omega,\qquad \dots,\qquad r^{(k)} = \omega_d^{(k-1)} - \omega^{(k-1)},
$$

and the rotational tracker is the scalar linear system $s^{(n)} = \sum_k K_k(t)\, s^{(k)}$ along $n$. This covers rest-to-rest motions toward a fixed orientation (initial angular velocity zero or along the error axis) and targets spinning about a fixed axis with any time profile $\sigma(t)$, including jumps in $\sigma$ or $\dot\sigma$.

In this case the implementation needs no Jacobians: $e=\theta n$ together with the plain differences $\omega_d-\omega$, $\dot\omega_d-\dot\omega,\dots$ is exact. Outside this case those differences are only first-order approximations of $\dot r, \ddot r,\dots$; for example $\dot r = (\omega_d-\omega) - \tfrac12\, r\times(\omega_d+\omega) + O(\|r\|^2\|\omega\|)$.

---

## 8. Numerical verification (summary)

Checks performed with explicit Euler, time step 1 ms:

- $\Phi^{-1}\circ\Phi$ is the identity to $10^{-15}$; $\omega$ from (V') and $\dot\omega$ from (A') match finite differences of the generated $q(t)$ and $\omega(t)$ to $10^{-10}$; $\dot J_r$ matches finite differences to $10^{-10}$.
- Generator form, online minimum-jerk toward a target tumbling with time-varying angular velocity, starting $150^\circ$ away with a non-zero initial angular velocity: the angle to the target follows $\|e(t)\|$, the generated $(q,\omega,\dot\omega)$ stream is self-consistent to $O(\Delta t)$, and the target is reached at the deadline with residual angle below $10^{-3}$ degrees and $\|\omega-\omega_d\|\approx 2\times10^{-3}$ rad/s.
- Controller form, spring-mass-damper with an initial angular velocity orthogonal to the error axis: deviation from the linear solution is $O(\Delta t)$ (0.08° at 1 ms, halving with the time step).
- Single-axis case (Section 7), with and without target velocity and pose jumps: the rotational and linear solutions agree to $10^{-12}$ degrees.

---

## Appendix: reference implementation of the maps (Python)

Quaternions are `[w, x, y, z]`; angular velocities are world-frame. Covers $\Phi$ and $\Phi^{-1}$ up to $(q,\omega,\dot\omega)\leftrightarrow(r,\dot r,\ddot r)$.

```python
import numpy as np

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def Exp(r):                       # rotation vector -> unit quaternion
    th = np.linalg.norm(r)
    return np.concatenate([[np.cos(th/2)], 0.5*np.sinc(th/(2*np.pi))*r])   # np.sinc(x) = sin(pi x)/(pi x)

def Log(q):                       # unit quaternion -> rotation vector, |r| <= pi
    if q[0] < 0: q = -q           # shortest rotation
    th = 2*np.arctan2(np.linalg.norm(q[1:]), q[0])
    return q[1:]*(2/np.sinc(th/(2*np.pi)))

def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def Rot(r):                       # rotation matrix of Exp(r)
    th, S = np.linalg.norm(r), hat(r)
    if th < 1e-8: return np.eye(3) + S
    return np.eye(3) + np.sin(th)/th*S + (1 - np.cos(th))/th**2*S@S

def _coef(th):                    # a, b and a'/th, b'/th  (series near 0)
    if th < 1e-4: return 0.5, 1/6, -1/12, -1/60
    c, s = np.cos(th), np.sin(th)
    a, b = (1 - c)/th**2, (th - s)/th**3
    da, db = s/th**2 - 2*(1 - c)/th**3, (1 - c)/th**3 - 3*(th - s)/th**4
    return a, b, da/th, db/th

def Jr(r):
    a, b, _, _ = _coef(np.linalg.norm(r)); S = hat(r)
    return np.eye(3) - a*S + b*S@S

def Jl(r):
    return Jr(-r)

def Jr_dot(r, rd):                # d/dt J_r(r(t))
    a, b, da_th, db_th = _coef(np.linalg.norm(r)); S, Sd = hat(r), hat(rd)
    k = r @ rd                    # = theta * theta'
    return -da_th*k*S - a*Sd + db_th*k*S@S + b*(Sd@S + S@Sd)

def to_chart(q, w, wdot, qd, wd, wddot):
    """Phi: tracker state (q, w, w') + target state -> chart state (r, r', r'')"""
    r = Log(qmul(qd, qconj(q))); RT = Rot(r).T; J = Jr(r)
    rd = np.linalg.solve(J, RT @ wd - w)                                      # (V)
    RT_dot = -RT @ hat(Jl(r) @ rd)
    rdd = np.linalg.solve(J, RT_dot @ wd + RT @ wddot - Jr_dot(r, rd) @ rd - wdot)   # (A)
    return r, rd, rdd

def from_chart(r, rd, rdd, qd, wd, wddot):
    """Phi^-1: chart state (r, r', r'') + target state -> tracker state (q, w, w')"""
    RT = Rot(r).T; J = Jr(r)
    q = qmul(Exp(-r), qd)
    w = RT @ wd - J @ rd                                                      # (V')
    RT_dot = -RT @ hat(Jl(r) @ rd)
    wdot = RT_dot @ wd + RT @ wddot - Jr_dot(r, rd) @ rd - J @ rdd            # (A')
    return q, w, wdot
```

Generator-form step for online minimum-jerk (state `r, rd, rdd`, target state `qd, wd, wddot` at time `t`):

```python
tau = T - t
if tau > tau_min:
    u = 60/tau**3*r + 36/tau**2*rd + 9/tau*rdd
    r, rd, rdd = r + rd*dt, rd + rdd*dt, rdd - u*dt
else:
    r, rd, rdd = 0*r, 0*rd, 0*rdd            # target reached: move with it
q, w, wdot = from_chart(r, rd, rdd, qd, wd, wddot)
```
