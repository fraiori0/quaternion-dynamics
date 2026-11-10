"""
rotdyn.py -- lifting linear error dynamics to SO(3) with a log-type orientation error.

Conventions
-----------
* Quaternions q = [w, x, y, z], Hamilton product, unit norm.
* Angular velocities are WORLD (spatial) frame:  dq/dt = 1/2 [0, w] (x) q.
  This is the frame consistent with the error q_e = q_des (x) q^*  (left error).
  If your w is body-frame, use q^* (x) q_des instead (right error) -- same results.
* Integration: Lie-Euler, q <- Exp(w dt) (x) q ; w <- w + a dt  (first order, like Euler).
"""
import numpy as np

I3 = np.eye(3)

# ----------------------------------------------------------------- quaternions
def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qexp(r):
    """rotation vector r = theta*n  ->  unit quaternion (cos th/2, n sin th/2)"""
    th = np.linalg.norm(r)
    return np.concatenate([[np.cos(th/2)], 0.5*np.sinc(th/(2*np.pi))*r])  # sin(th/2)/th * r

def qlog(q):
    """unit quaternion -> rotation vector theta*n, shortest path (theta in [0, pi])"""
    if q[0] < 0: q = -q
    v = q[1:]; s = np.linalg.norm(v)
    th = 2*np.arctan2(s, q[0])
    return v * (2.0/np.sinc(th/(2*np.pi)))                         # = th * v/|v|

def qnormalize(q):
    return q/np.linalg.norm(q)

# ------------------------------------------------------- orientation errors
def err_user_literal(q, qd):
    """Exactly the LaTeX as written, with sinc read as numpy's normalized sinc
    (np.sinc(x) = sin(pi x)/(pi x)), which is the only reading that makes
    the 2*pi meaningful.  No hemisphere check, as in the LaTeX."""
    pq = qmul(qconj(q), qd)
    d = 2*np.arctan2(np.linalg.norm(pq[1:]), pq[0])
    rho = 1.0/np.sinc(d/(2*np.pi))
    return rho*qmul(qd, qconj(q))[1:]

def err_full(q, qd):   # theta * n   (rotation vector, = 2 log q_e)
    return qlog(qmul(qd, qconj(q)))

def err_half(q, qd):   # theta/2 * n (quaternion log, = what the LaTeX gives)
    return 0.5*err_full(q, qd)

def err_vec(q, qd):    # sin(theta/2) n, classic quaternion vector-part error (rho = 1)
    qe = qmul(qd, qconj(q))
    if qe[0] < 0: qe = -qe
    return qe[1:]

ERRORS = {"full": err_full, "half": err_half, "vec": err_vec}

# --------------------------------------------------------- SO(3) Jacobians
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def Rmat(r):
    th = np.linalg.norm(r); S = skew(r)
    if th < 1e-9: return I3 + S
    return I3 + np.sin(th)/th*S + (1-np.cos(th))/th**2*S@S

def Jl(r):
    th = np.linalg.norm(r); S = skew(r)
    if th < 1e-6: return I3 + 0.5*S
    return I3 + (1-np.cos(th))/th**2*S + (th-np.sin(th))/th**3*S@S

def Jl_inv(r):
    th = np.linalg.norm(r); S = skew(r)
    if th < 1e-6: return I3 - 0.5*S
    return I3 - 0.5*S + (1/th**2 - (1+np.cos(th))/(2*th*np.sin(th)))*S@S

def Jr(r):      return Jl(-r)
def Jr_inv(r):  return Jl_inv(-r)

def rdot(r, w, wd):
    """exact time derivative of r = Log(qd (x) q^*) given world-frame w, wd"""
    return Jl_inv(r) @ (wd - Rmat(r) @ w)

# --------------------------------------------------------------- targets
def static_target(qd0):
    def gen(t, dt, state):
        return state, np.zeros(3), np.zeros(3), np.zeros(3)
    return gen

def moving_target(wfun, afun, jfun=None):
    """target integrated from a prescribed world angular velocity profile"""
    def gen(t, dt, qd):
        wd, ad = wfun(t), afun(t)
        jd = jfun(t) if jfun else np.zeros(3)
        return qd, wd, ad, jd
    return gen

# ------------------------------------------------------------ simulators
def sim2(q0, w0, qd0, law, err="full", T=5.0, dt=1e-3, target=None, exact=False):
    """Second-order virtual system  dw/dt = a.
    naive:  a = a_d + u(e, w_d - w, t)          with e from ERRORS[err]
    exact:  a chosen so that r'' = -u(r, r', t) exactly (feedback linearization)
    law(e, ed, t) returns u such that linear error dynamics are e'' = -u."""
    N = int(round(T/dt)); q, w, qd = q0.copy(), w0.copy(), qd0.copy()
    target = target or static_target(qd0)
    out = {k: np.zeros((N+1, 3)) for k in ("e", "w", "a", "r", "wd")}
    out["th"] = np.zeros(N+1); out["t"] = np.arange(N+1)*dt; out["cost_u"] = np.zeros(N+1)
    efun = ERRORS[err]
    for k in range(N+1):
        t = k*dt
        qd, wd, ad, _ = target(t, dt, qd)
        r = err_full(q, qd)
        if not exact:
            e = efun(q, qd); ed = wd - w
            u = law(e, ed, t)
            a = ad + u
        else:
            e = r; ed = rdot(r, w, wd)
            v = -law(e, ed, t)                      # desired r''
            h = 1e-6                                # directional FD for (d f/d r) r'
            fr = (rdot(r + h*ed, w, wd) - rdot(r - h*ed, w, wd))/(2*h)
            a = Jr(r) @ (fr + Jl_inv(r) @ ad - v)
        out["e"][k], out["w"][k], out["a"][k], out["r"][k], out["wd"][k] = e, w, a, r, wd
        out["th"][k] = np.linalg.norm(r)
        if k == N: break
        q = qnormalize(qmul(qexp(w*dt), q)); w = w + a*dt
        qd = qnormalize(qmul(qexp(wd*dt), qd))
    return out

def sim3(q0, w0, a0, qd0, law, err="full", T=2.0, dt=1e-3, tau_stop=0.02,
         goal_switch=None, Tf=None):
    """Third-order virtual system  da/dt = j,  j = j_d + u(e, w_d-w, a_d-a, t).
    Static target(s). goal_switch = (t_s, qd_new, T_new) changes goal/deadline online."""
    N = int(round(T/dt)); q, w, a, qd = q0.copy(), w0.copy(), a0.copy(), qd0.copy()
    Tf = T if Tf is None else Tf; efun = ERRORS[err]
    out = {k: np.zeros((N+1, 3)) for k in ("e", "w", "a", "r")}
    out["th"] = np.zeros(N+1); out["t"] = np.arange(N+1)*dt
    frozen = False
    for k in range(N+1):
        t = k*dt
        if goal_switch and abs(t - goal_switch[0]) < dt/2:
            qd, Tf = goal_switch[1], goal_switch[2]
        tau = Tf - t
        r = err_full(q, qd); e = efun(q, qd)
        if tau < tau_stop and not frozen:
            frozen = True; w = np.zeros(3); a = np.zeros(3)   # hold pose at the deadline
        jj = np.zeros(3) if frozen else law(e, -w, -a, tau)
        out["e"][k], out["w"][k], out["a"][k], out["r"][k] = e, w, a, r
        out["th"][k] = np.linalg.norm(r)
        if k == N: break
        if not frozen:
            q = qnormalize(qmul(qexp(w*dt), q)); w = w + a*dt; a = a + jj*dt
    return out

def lin2(e0, ed0, law, T=5.0, dt=1e-3):
    """linear reference e'' = -u(e, e', t), same Euler scheme"""
    N = int(round(T/dt)); e, ed = e0.copy(), ed0.copy()
    E = np.zeros((N+1, 3)); Ed = np.zeros((N+1, 3)); U = np.zeros((N+1, 3))
    for k in range(N+1):
        u = law(e, ed, k*dt); E[k], Ed[k], U[k] = e, ed, u
        if k == N: break
        e, ed = e + ed*dt, ed - u*dt
    return E, Ed, U

def lin3(e0, ed0, edd0, law, T=2.0, dt=1e-3, tau_stop=0.02, Tf=None):
    N = int(round(T/dt)); e, ed, edd = e0.copy(), ed0.copy(), edd0.copy()
    Tf = T if Tf is None else Tf
    E = np.zeros((N+1, 3)); frozen = False
    for k in range(N+1):
        tau = Tf - k*dt
        if tau < tau_stop: frozen = True
        E[k] = e
        if k == N: break
        if not frozen:
            u = law(e, ed, edd, tau)
            e, ed, edd = e + ed*dt, ed + edd*dt, edd - u*dt
    return E

# ------------------------------------------------------------- linear laws
def smd_law(K, D):
    K = np.atleast_2d(K) if np.ndim(K) == 2 else K*I3
    D = np.atleast_2d(D) if np.ndim(D) == 2 else D*I3
    return lambda e, ed, t: K @ e + D @ ed

def lqr_double_integrator(qp, qv, r):
    """Continuous LQR for e'' = -u, cost int qp e^2 + qv e'^2 + r u^2 (per axis).
    Closed form: kp = sqrt(qp/r), kd = sqrt(qv/r + 2 kp)."""
    kp = np.sqrt(qp/r); kd = np.sqrt(qv/r + 2*kp)
    P = np.array([[np.sqrt(qp*(qv + 2*np.sqrt(qp*r))), np.sqrt(qp*r)],
                  [np.sqrt(qp*r), np.sqrt(r*(qv + 2*np.sqrt(qp*r)))]])
    return kp, kd, P

def minjerk_law():
    """Hoff & Arbib online minimum-jerk feedback, tau = time-to-go.
    In error form (static goal): e''' = -u,  u = 60/tau^3 e + 36/tau^2 e' + 9/tau e''."""
    return lambda e, ed, edd, tau: 60/tau**3*e + 36/tau**2*ed + 9/tau*edd

def minjerk_profile(th0, T, t):
    s = np.clip(t/T, 0, 1)
    p = 10*s**3 - 15*s**4 + 6*s**5
    dp = (30*s**2 - 60*s**3 + 30*s**4)/T
    ddp = (60*s - 180*s**2 + 120*s**3)/T**2
    return th0*(1-p), th0*dp, th0*ddp

# ------------------------------------------------ exact (chart) formulation
def _ab(th):
    """J_r(r) = I - a S + b S^2,  S = [r]x ; returns a, b, a', b'"""
    if th < 1e-4:
        return 0.5 - th**2/24, 1/6 - th**2/120, -th/12, -th/60
    c, s = np.cos(th), np.sin(th)
    return ((1-c)/th**2, (th-s)/th**3,
            s/th**2 - 2*(1-c)/th**3, (1-c)/th**3 - 3*(th-s)/th**4)

def Jr_dot(r, rd):
    """d/dt J_r(r(t)) for given r, r'"""
    th = np.linalg.norm(r); a, b, da, db = _ab(th)
    S, Sd = skew(r), skew(rd)
    thd_over = (r @ rd)                       # theta * theta'  (avoid /theta)
    # a' theta' = (da/th) * (r.r') ; use series-safe ratio
    ra = da/th if th > 1e-4 else -1/12
    rb = db/th if th > 1e-4 else -1/60
    return -ra*thd_over*S - a*Sd + rb*thd_over*S@S + b*(Sd@S + S@Sd)

def chart_to_rot(r, rd, rdd, qd, wd=np.zeros(3), ad=np.zeros(3)):
    """(r, r', r'') with r = Log(qd q*)  ->  (q, w, w') ; world-frame rates."""
    R = Rmat(r); J = Jr(r)
    q = qmul(qexp(-r), qd)
    w = R.T @ wd - J @ rd
    Rd_T = -R.T @ skew(Jl(r) @ rd)            # d/dt R(r)^T
    a = Rd_T @ wd + R.T @ ad - Jr_dot(r, rd) @ rd - J @ rdd
    return q, w, a

def rot_to_chart(q, w, a, qd, wd=np.zeros(3), ad=np.zeros(3)):
    """inverse map (q, w, w') -> (r, r', r''); used to (re)initialize on a goal change."""
    r = err_full(q, qd); R = Rmat(r); J = Jr(r)
    rd = np.linalg.solve(J, R.T @ wd - w)
    Rd_T = -R.T @ skew(Jl(r) @ rd)
    rdd = np.linalg.solve(J, Rd_T @ wd + R.T @ ad - Jr_dot(r, rd) @ rd - a)
    return r, rd, rdd
