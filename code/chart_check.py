"""Exact (chart) formulation: checks + comparison with the intrinsic (naive) form."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rotdyn import *
OUT = "/mnt/user-data/outputs/"; rng = np.random.default_rng(2); deg = np.degrees
unit = lambda v: v/np.linalg.norm(v)

# --- 1. Jr_dot and the maps, against finite differences
r, rd, rdd = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3); h = 1e-6
fd = (Jr(r + h*rd) - Jr(r - h*rd))/(2*h)
print("Jr_dot vs FD:", np.abs(fd - Jr_dot(r, rd)).max())
qd = qexp(rng.normal(size=3)); wd = rng.normal(size=3); ad = rng.normal(size=3)
r = unit(r)*2.0
q, w, a = chart_to_rot(r, rd, rdd, qd, wd, ad)
print("round trip chart->rot->chart:", max(np.abs(x - y).max() for x, y in zip((r, rd, rdd), rot_to_chart(q, w, a, qd, wd, ad))))
# consistency: w must be the true angular velocity of q(t), a its derivative
def traj(t):  # r(t) quadratic, qd(t) rotating with wd + ad t
    rt = r + rd*t + 0.5*rdd*t**2; qdt = qmul(qexp(wd*t + 0.5*ad*t**2), qd)  # approx for small t
    return rt, rd + rdd*t, qdt
dt = 1e-5
(r1, rd1, qd1), (r2, rd2, qd2) = traj(-dt), traj(dt)
q1 = qmul(qexp(-r1), qd1); q2 = qmul(qexp(-r2), qd2)
w_fd = qlog(qmul(q2, qconj(q1)))/(2*dt)
print("w from map vs FD of q(t):", np.abs(w_fd - w).max())

# --- 2. min-jerk generator in chart form, with an off-axis goal switch
dt, T0, TS, T1, Tend = 1e-3, 2.0, 0.8, 2.5, 3.0
n0 = unit(np.array([1., -2., .5])); perp = unit(np.cross(n0, [0, 0, 1.]))
qd0 = qexp(np.array([.3, .2, -.4])); qd1 = qmul(qexp(np.radians(90)*perp), qd0)
q0 = qmul(qexp(-np.radians(170)*n0), qd0)
law = minjerk_law()

def chart_minjerk():
    qd, Tf = qd0, T0
    x, xd, xdd = rot_to_chart(q0, np.zeros(3), np.zeros(3), qd)
    Q, W, A, TH = [], [], [], []; frozen = False
    for k in range(int(Tend/dt) + 1):
        t = k*dt
        if abs(t - TS) < dt/2:                                  # goal change: re-initialize chart state
            q, w, a = chart_to_rot(x, xd, xdd, qd)
            qd, Tf = qd1, T1
            x, xd, xdd = rot_to_chart(q, w, a, qd)
        q, w, a = chart_to_rot(x, xd, xdd, qd)
        Q.append(q); W.append(w); A.append(a); TH.append(np.linalg.norm(x))
        tau = Tf - t
        if tau < 0.02: frozen = True; xd[:] = 0; xdd[:] = 0
        if not frozen:
            u = law(x, xd, xdd, tau)
            x, xd, xdd = x + xd*dt, xd + xdd*dt, xdd - u*dt
    return np.array(Q), np.array(W), np.array(A), np.array(TH)

Qc, Wc, Ac, THc = chart_minjerk()
nv = sim3(q0, np.zeros(3), np.zeros(3), qd0, law, err="full", T=Tend, Tf=T0, goal_switch=(TS, qd1, T1))
t = np.arange(len(THc))*dt; ks = int(TS/dt)
print(f"chart: |w|,|a| jump at switch: {np.linalg.norm(Wc[ks+1]-Wc[ks]):.2e}, {np.linalg.norm(Ac[ks+1]-Ac[ks]):.2e}")
print(f"chart: angle at deadline {deg(THc[int((T1-0.02)/dt)]):.2e} deg")
# consistency of the generated stream: q integrates w, w integrates a
wq = np.array([qlog(qmul(Qc[k+1], qconj(Qc[k])))/dt for k in range(len(Qc)-1)])
sep = np.array([deg(np.linalg.norm(qlog(qmul(Qc[k], qconj(qmul(qexp(-nv['r'][k]), qd0 if t[k] < TS else qd1)))))) for k in range(len(t))])
print(f"max orientation gap chart vs naive: {sep.max():.2f} deg")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(t, deg(THc), label="exact (chart)"); ax[0].plot(t, deg(nv["th"]), "--", label="intrinsic (your form, fixed)")
ax[0].axvline(TS, c="gray", ls=":"); ax[0].set(xlabel="t [s]", ylabel="angle to current goal [deg]", title="min-jerk with goal switch"); ax[0].legend(fontsize=8)
ax[1].plot(t, np.linalg.norm(Wc, axis=1), label=r"$|\omega|$ chart"); ax[1].plot(t, np.linalg.norm(nv["w"], axis=1), "--", label=r"$|\omega|$ intrinsic")
ax[1].axvline(TS, c="gray", ls=":"); ax[1].set(xlabel="t [s]", ylabel="rad/s", title="speed"); ax[1].legend(fontsize=8)
ax[2].plot(t, sep); ax[2].set(xlabel="t [s]", ylabel="deg", title="orientation gap between the two")
plt.tight_layout(); plt.savefig(OUT + "fig6_chart_vs_intrinsic.png", dpi=130)
dd = np.linalg.norm(np.diff(Wc, axis=0)/dt - Ac[:-1], axis=1)
kf = int((T1 - 0.02)/dt) - 2
print("stream consistency before freeze: max|dq/dt - w| =", np.abs(wq[:kf]-Wc[:kf]).max(), " max|dw/dt - a| =", dd[:kf].max())
