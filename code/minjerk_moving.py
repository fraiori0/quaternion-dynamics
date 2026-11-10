"""Online min-jerk tracking of a TUMBLING target (axis not fixed), full target derivatives used.
Linear reference (position analog, full feedforward):  e''' = -(60/tau^3 e + 36/tau^2 e' + 9/tau e'')
 (I) intrinsic: w'' = w_d'' + 60/tau^3 r + 36/tau^2 (w_d - w) + 9/tau (w_d' - w')
 (II) exact, generator form: propagate (r, r', r'') with the linear ODE, output
      q = Exp(-r) qd(t),  w = R^T w_d - J_r r',  w' = (d/dt R^T) w_d + R^T w_d' - J_r' r' - J_r r''
"""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rotdyn import *
OUT = "/mnt/user-data/outputs/"; deg = np.degrees; unit = lambda v: v/np.linalg.norm(v)
dt, T, Tend = 1e-3, 2.5, 3.0
wf = lambda t: np.array([0.8*np.sin(0.7*t), 0.6*np.cos(0.4*t), 0.5])
af = lambda t: np.array([0.56*np.cos(0.7*t), -0.24*np.sin(0.4*t), 0.0])
jf = lambda t: np.array([-0.392*np.sin(0.7*t), -0.096*np.cos(0.4*t), 0.0])
law = minjerk_law(); N = int(Tend/dt)
qd0 = qexp(np.array([.3, .2, -.4])); q0 = qmul(qexp(-np.radians(150)*unit(np.array([0., 1, 1]))), qd0)
w0 = np.array([0.5, 0., -0.5]); a0 = np.zeros(3)                  # tracker already moving

QD = [qd0]                                                         # target trajectory
for k in range(N): QD.append(qnormalize(qmul(qexp(wf(k*dt)*dt), QD[-1])))

def run(mode):
    TH, Q, W, A = [], [], [], []; frozen = False
    if mode == "II": x, xd, xdd = rot_to_chart(q0, w0, a0, qd0, wf(0), af(0))
    else: q, w, a = q0.copy(), w0.copy(), a0.copy()
    for k in range(N+1):
        t = k*dt; tau = T - t; qd, wd, ad, jd = QD[k], wf(t), af(t), jf(t)
        if mode == "II":
            q, w, a = chart_to_rot(x, xd, xdd, qd, wd, ad); r = x
        else:
            r = err_full(q, qd)
        TH.append(np.linalg.norm(r)); Q.append(q); W.append(w.copy()); A.append(a.copy())
        if tau < 0.02: frozen = True
        if k == N: break
        if mode == "II":
            u = np.zeros(3) if frozen else law(x, xd, xdd, tau)
            x, xd, xdd = x + xd*dt, xd + xdd*dt, xdd - u*dt
            if frozen: xd[:] = 0; xdd[:] = 0                         # then: ride along with the target
        else:
            j = jd + (np.zeros(3) if frozen else law(r, wd - w, ad - a, tau))
            if frozen: w, a = wd.copy(), ad.copy()
            q = qnormalize(qmul(qexp(w*dt), q)); w = w + a*dt; a = a + j*dt
    return np.array(TH), np.array(Q), np.array(W), np.array(A)

out = {m: run(m) for m in ("I", "II")}
t = np.arange(N+1)*dt
r0 = err_full(q0, qd0)
twin_I = lin3(r0, wf(0) - w0, af(0) - a0, law, T=Tend, Tf=T)
_, rd0, rdd0 = rot_to_chart(q0, w0, a0, qd0, wf(0), af(0))
twin_II = lin3(r0, rd0, rdd0, law, T=Tend, Tf=T)
kT = int((T - 0.02)/dt)
for m, tw in (("I", twin_I), ("II", twin_II)):
    th = out[m][0]
    print(f"({m}) max | angle - |e_twin| | = {deg(np.max(np.abs(th[:kT] - np.linalg.norm(tw[:kT], axis=1)))):.3f} deg ; angle at deadline {deg(th[kT]):.1e} deg")
# the generated (q, w, w') stream of (II) must be self-consistent: q integrates w, w integrates w'
Q, W, A = out["II"][1:]
wq = np.array([qlog(qmul(Q[k+1], qconj(Q[k])))/dt for k in range(kT)])
print("(II) stream consistency: max|dq/dt - w| =", np.abs(wq - W[:kT]).max().round(4),
      " max|dw/dt - w'| =", np.abs(np.diff(W[:kT+1], axis=0)/dt - A[:kT]).max().round(4))
print("(II) relative angular velocity at deadline |w - w_d| =", np.linalg.norm(W[kT] - wf(kT*dt)).round(5))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(t, deg(np.linalg.norm(twin_II, axis=1)), "k", lw=3, alpha=.35, label="linear twin (II)")
ax[0].plot(t, deg(out["II"][0]), label="(II) exact"); ax[0].plot(t, deg(out["I"][0]), "--", label="(I) intrinsic")
ax[0].set(xlabel="t [s]", ylabel="angle to moving target [deg]", title="online min-jerk, tumbling target, T=2.5 s"); ax[0].legend(fontsize=8)
ax[1].plot(t, deg(np.abs(out["II"][0] - np.linalg.norm(twin_II, axis=1))) + 1e-12, label="(II)")
ax[1].plot(t, deg(np.abs(out["I"][0] - np.linalg.norm(twin_I, axis=1))) + 1e-12, "--", label="(I)")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-14, 100); ax[1].set(xlabel="t [s]", ylabel="deg", title="| angle − twin |"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig8_minjerk_tumbling.png", dpi=130)
