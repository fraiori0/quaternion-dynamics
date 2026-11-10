"""Moving targets with NO acceleration feedforward: does the rotational tracker
behave like a position tracker chasing a moving point?  Linear twin:
    e'' = -Kp e - Kd e' + a_d(t)        (a_d = target acceleration, enters as a disturbance)
Rotational trackers compared:
  (I)   intrinsic : w' = Kp r + Kd (w_d - w)                       [your form, fixed]
  (C)   coupling  : w' = J_r ( h - v ),  v = -Kp r - Kd r'          -> r'' = v + J_l^{-1}(r) w_d'
  (D)   exact     : w' = J_r ( h + (J_l^{-1} - I) w_d' - v )        -> r'' = v + w_d'
  with r' = J_l^{-1}(r) w_d - J_r^{-1}(r) w  and  h = (d/dr)[J_l^{-1} w_d - J_r^{-1} w] . r'
"""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rotdyn import *
OUT = "/mnt/user-data/outputs/"; deg = np.degrees; unit = lambda v: v/np.linalg.norm(v)
import sys
dt = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
kp, kd, _ = lqr_double_integrator(4.0, 0.5, 0.1)          # LQR gains, as before

def f(r, w, wd): return Jl_inv(r) @ wd - Jr_inv(r) @ w     # = r'

def track(q0, w0, qd0, target, T, mode):
    """target(t) -> (w_d, a_d_smooth, dw_bump, dq_bump) ; bumps are applied at the step they occur"""
    N = int(round(T/dt)); q, w, qd = q0.copy(), w0.copy(), qd0.copy()
    wd = target(0)[0].copy()
    R_, TH, W, A = np.zeros((N+1, 3)), np.zeros(N+1), np.zeros((N+1, 3)), np.zeros((N+1, 3))
    for k in range(N+1):
        t = k*dt
        _, ad, dwb, dqb = target(t)
        if dqb is not None: qd = qmul(dqb, qd)               # pose bump of the target
        if dwb is not None: wd = wd + dwb                    # velocity bump of the target
        imp = dwb/dt if dwb is not None else 0              # ... i.e. an impulse in w_d'
        r = err_full(q, qd)
        if mode == "I":
            a = kp*r + kd*(wd - w)
        else:
            rd = f(r, w, wd); v = -kp*r - kd*rd; hh = 1e-6
            h = (f(r + hh*rd, w, wd) - f(r - hh*rd, w, wd))/(2*hh)
            a = Jr(r) @ (h - v + ((Jl_inv(r) - I3) @ (ad + imp) if mode == "D" else 0))
        R_[k], TH[k], W[k], A[k] = r, np.linalg.norm(r), w, a
        if k == N: break
        q = qnormalize(qmul(qexp(w*dt), q)); w = w + a*dt
        qd = qnormalize(qmul(qexp(wd*dt), qd)); wd = wd + ad*dt
    return dict(r=R_, th=TH, w=W, a=A, t=np.arange(N+1)*dt)

def twin(e0, ed0, target, T, de_jump=None):
    """linear twin; velocity bumps add to e', pose bumps add de_jump(t) to e"""
    N = int(round(T/dt)); e, ed = e0.copy(), ed0.copy(); E = np.zeros((N+1, 3))
    for k in range(N+1):
        t = k*dt; _, ad, dwb, dqb = target(t)
        if dqb is not None: e = e + de_jump(dqb)
        if dwb is not None: ed = ed + dwb
        E[k] = e
        if k == N: break
        e, ed = e + ed*dt, ed + (-kp*e - kd*ed + ad)*dt
    return E

def at(t, t0): return abs(t - t0) < dt/2

# ------------------------------------------------ E1: fixed-axis target, arbitrary sigma(t) + bumps
n = unit(np.array([1., -2., .5])); qbar = qexp(np.array([.3, .2, -.4]))
def tgt1(t):
    wd = (0.5 + 0.39*np.cos(1.3*t))*n; ad = -0.507*np.sin(1.3*t)*n
    return wd, ad, (1.2*n if at(t, 4) else None), (qexp(np.radians(40)*n) if at(t, 8) else None)
q0 = qmul(qexp(-np.radians(80)*n), qbar)                  # 80 deg behind, at rest
T1 = 12
o1 = track(q0, np.zeros(3), qbar, tgt1, T1, "I")
E1 = twin(o1["r"][0], tgt1(0)[0], tgt1, T1, de_jump=lambda dq: qlog(dq))
print("E1 fixed-axis target with velocity+pose bumps, intrinsic form, no feedforward:")
print(f"   max |r - e_twin| = {deg(np.max(np.linalg.norm(o1['r'] - E1, axis=1))):.1e} deg")

# ------------------------------------------------ E2: general tumbling, constant w_d
wc = np.array([0.4, -0.7, 0.9]); m = unit(np.array([0., 1., 1.]))
tgt2 = lambda t: (wc, np.zeros(3), None, None)
q0b = qmul(qexp(-np.radians(90)*m), qbar)
print("E2 constant-velocity tumbling, axis not aligned with error, no feedforward:")
for md in ("I", "C"):
    o = track(q0b, np.zeros(3), qbar, tgt2, 8, md)
    ed0 = wc if md == "I" else f(o["r"][0], np.zeros(3), wc)
    E = twin(o["r"][0], ed0, tgt2, 8)
    print(f"   {md}: max |r - e_twin| = {deg(np.max(np.linalg.norm(o['r'] - E, axis=1))):.3f} deg ; max | |r|-|e| | = {deg(np.max(np.abs(o['th'] - np.linalg.norm(E, axis=1)))):.3f} deg")

# ------------------------------------------------ E3: general tumbling, time-varying w_d, bumps
def tgt3(t):
    wd = np.array([0.8*np.sin(0.7*t), 0.6*np.cos(0.4*t), 0.5])
    ad = np.array([0.56*np.cos(0.7*t), -0.24*np.sin(0.4*t), 0.0])
    return wd, ad, (np.array([0., 1.5, -1.0]) if at(t, 5) else None), None
T3 = 10
res, twins = {}, {}
print("E3 tumbling target (time-varying w_d) + velocity bump at t=5 s, no feedforward:")
for md in ("I", "C", "D"):
    o = track(q0b, np.zeros(3), qbar, tgt3, T3, md); res[md] = o
    ed0 = tgt3(0)[0] if md == "I" else f(o["r"][0], np.zeros(3), tgt3(0)[0])
    twins[md] = twin(o["r"][0], ed0, tgt3, T3)
    dv = np.linalg.norm(o["r"] - twins[md], axis=1)
    print(f"   {md}: max |r - e_twin| = {deg(dv.max()):7.3f} deg ; max | |r|-|e| | = {deg(np.max(np.abs(o['th'] - np.linalg.norm(twins[md], axis=1)))):6.3f} deg ; max |w'| = {np.abs(np.linalg.norm(o['a'],axis=1)).max():8.1f} rad/s^2")

t = res["I"]["t"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(o1["t"], deg(np.linalg.norm(E1, axis=1)), "k", lw=3, alpha=.35, label="linear twin")
ax[0].plot(o1["t"], deg(o1["th"]), label="rotation, intrinsic form")
for tb in (4, 8): ax[0].axvline(tb, c="gray", ls=":")
ax[0].set(xlabel="t [s]", ylabel="tracking error [deg]", title="fixed-axis target: velocity bump (4 s), pose bump (8 s)"); ax[0].legend(fontsize=8)
ax[1].plot(t, deg(np.linalg.norm(twins["C"], axis=1)), "k", lw=3, alpha=.35, label="linear twin")
for md, st in zip(("I", "C", "D"), ("-", "--", ":")): ax[1].plot(t, deg(res[md]["th"]), st, label=md)
ax[1].axvline(5, c="gray", ls=":"); ax[1].set(xlabel="t [s]", ylabel="tracking error [deg]", title="tumbling target, velocity bump at 5 s"); ax[1].legend(fontsize=8)
for md, st in zip(("I", "C", "D"), ("-", "--", ":")):
    ax[2].semilogy(t, deg(np.linalg.norm(res[md]["r"] - twins[md], axis=1)) + 1e-9, st, label=md)
ax[2].set(xlabel="t [s]", ylabel="deg", title=r"$|r - e_{twin}|$"); ax[2].set_ylim(1e-6, 30); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig7_moving_target.png", dpi=130)
