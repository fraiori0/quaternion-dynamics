import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rotdyn import *

OUT = "/mnt/user-data/outputs/"
rng = np.random.default_rng(0)
def unit(v): return v/np.linalg.norm(v)
def deg(x): return np.degrees(x)
print_ = lambda *a: print(*a, flush=True)

# =========================================================== 0. sanity checks
print_("== 0. sanity checks")
qd = qexp(unit(np.array([1., 2., -1]))*np.radians(140))
q = qexp(unit(np.array([0., 1., 1]))*np.radians(10))
th = np.linalg.norm(err_full(q, qd))
print_(f"angle {deg(th):.2f} deg | |user literal| = {deg(np.linalg.norm(err_user_literal(q,qd))):.2f} deg"
       f" | |full| = {deg(np.linalg.norm(err_full(q,qd))):.2f} | |half| = {deg(np.linalg.norm(err_half(q,qd))):.2f}")
print_("user literal == half ?", np.allclose(err_user_literal(q, qd), err_half(q, qd)))
# hemisphere issue: same rotation, quaternion sign flipped
print_("user literal with -qd (same rotation): |e| =",
       f"{deg(np.linalg.norm(err_user_literal(q, -qd))):.2f} deg (long way round, d > pi)")

# Jacobian / derivative-of-log formula check by finite differences
w = rng.normal(size=3); wd = rng.normal(size=3); dt = 1e-6
r0 = err_full(q, qd)
r1 = err_full(qmul(qexp(w*dt), q), qmul(qexp(wd*dt), qd))
fd = (r1 - r0)/dt
print_("r' finite diff vs Jl^-1 (wd - R w):", np.max(np.abs(fd - rdot(r0, w, wd))))
print_("naive (wd - w) error vs true r':   ", np.max(np.abs(fd - (wd - w))))
print_("first-order formula (wd-w) - 1/2 r x (wd+w) error:",
       np.max(np.abs(fd - ((wd - w) - 0.5*np.cross(r0, wd + w)))), "(O(|r|^2 |w|))")

# ======================================================= 1. spring-mass-damper
print_("\n== 1. spring-mass-damper, rest-to-rest, 150 deg, zeta = 0.3")
wn, zeta = 2*np.pi*0.5, 0.3
law = smd_law(wn**2, 2*zeta*wn)
n0 = unit(np.array([1., -2., 0.5])); th0 = np.radians(150)
qd0 = qexp(np.array([0.3, 0.2, -0.4]))
q0 = qmul(qexp(-th0*n0), qd0)                  # so that Log(qd q*) = th0 n0
T = 6.0
E, Ed, U = lin2(th0*n0, np.zeros(3), law, T=T)
res1 = {v: sim2(q0, np.zeros(3), qd0, law, err=v, T=T) for v in ("full", "half", "vec")}
t = res1["full"]["t"]
for v, o in res1.items():
    s = o["r"] @ n0
    drift = np.max(np.linalg.norm(o["r"] - np.outer(s, n0), axis=1))
    print_(f"  {v:5s}: max |angle - linear| = {deg(np.max(np.abs(s - E@n0))):8.4f} deg ;"
           f" max |w - lin| = {np.max(np.linalg.norm(o['w'] + Ed, axis=1)):.2e} rad/s ;"
           f" axis drift = {drift:.1e}")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(t, deg(E@n0), "k", lw=3, alpha=.35, label="linear reference")
for v, st in zip(res1, ("-", "--", ":")):
    ax[0].plot(t, deg(res1[v]["r"]@n0), st, label={"full": r"$e=\theta n$ (rot. vector)",
                "half": r"$e=\frac{\theta}{2}n$ (LaTeX as written)", "vec": r"$e=\sin\frac{\theta}{2}n$ ($\rho=1$)"}[v])
ax[0].set(xlabel="t [s]", ylabel="signed angle to target [deg]", title="Spring-mass-damper, 150° rest-to-rest"); ax[0].legend(fontsize=8)
ax[1].plot(t, np.linalg.norm(Ed, axis=1), "k", lw=3, alpha=.35, label="|linear velocity|")
ax[1].plot(t, np.linalg.norm(res1["full"]["w"], axis=1), label=r"$|\omega|$, $e=\theta n$")
ax[1].set(xlabel="t [s]", ylabel="rad/s", title="speed profile"); ax[1].legend(fontsize=8)
ax[2].semilogy(t, np.abs(deg(res1["full"]["r"]@n0 - E@n0)) + 1e-12)
ax[2].set(xlabel="t [s]", ylabel="deg", title=r"|angle − linear|, $e=\theta n$ (Euler, dt=1 ms)")
plt.tight_layout(); plt.savefig(OUT + "fig1_smd.png", dpi=130); plt.close()

# ============================================== 2. off-geodesic initial velocity
print_("\n== 2. SMD with initial angular velocity orthogonal to the error axis")
perp = unit(np.cross(n0, [0, 0, 1.]))
w0 = 1.5*perp
on = sim2(q0, w0, qd0, law, err="full", T=T)
ex = sim2(q0, w0, qd0, law, T=T, exact=True)
Eon, _, _ = lin2(th0*n0, -w0, law, T=T)                         # linear twin of naive
r0d = rdot(th0*n0, w0, np.zeros(3))
Eex, _, _ = lin2(th0*n0, r0d, law, T=T)                         # linear twin of exact
dev_on = np.max(np.abs(on["th"] - np.linalg.norm(Eon, axis=1)))
dev_ex = np.max(np.linalg.norm(ex["r"] - Eex, axis=1))
print_(f"  naive (full log + w_d - w): max | |r| - |x_lin| | = {deg(dev_on):.2f} deg")
print_(f"  exact (feedback-linearized): max |r - x_lin|     = {deg(dev_ex):.4f} deg")

# sweep
ths = np.radians([20, 45, 90, 135, 170]); wps = np.linspace(0, 3, 13)
sweep = np.zeros((len(ths), len(wps)))
for i, th_ in enumerate(ths):
    q0s = qmul(qexp(-th_*n0), qd0)
    for j, wp in enumerate(wps):
        o = sim2(q0s, wp*perp, qd0, law, err="full", T=T, dt=2e-3)
        El, _, _ = lin2(th_*n0, -wp*perp, law, T=T, dt=2e-3)
        sweep[i, j] = deg(np.max(np.abs(o["th"] - np.linalg.norm(El, axis=1))))
print_("  sweep (rows theta0 =", deg(ths), "deg; cols |w_perp| 0..3 rad/s):")
print_(np.array2string(sweep[:, ::3], precision=2))

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(t, deg(np.linalg.norm(Eon, axis=1)), "k", lw=3, alpha=.35, label="linear twin (naive)")
ax[0].plot(t, deg(on["th"]), label=r"naive: $e=\theta n$, $\dot e=\omega_d-\omega$")
ax[0].plot(t, deg(np.linalg.norm(Eex, axis=1)), "C3", lw=3, alpha=.35, label="linear twin (exact)")
ax[0].plot(t, deg(ex["th"]), "C3--", label=r"exact: $\dot r=J_l^{-1}(\omega_d-R\omega)$")
ax[0].set(xlabel="t [s]", ylabel="angle to target [deg]", title=r"SMD, 150°, $\omega_0\perp$ axis, 1.5 rad/s"); ax[0].legend(fontsize=7)
ax[1].plot(t, deg(on["th"] - np.linalg.norm(Eon, axis=1)), label="naive")
ax[1].plot(t, deg(ex["th"] - np.linalg.norm(Eex, axis=1)), "C3", label="exact")
ax[1].set(xlabel="t [s]", ylabel="deg", title="angle − linear twin"); ax[1].legend(fontsize=8)
for i, th_ in enumerate(ths):
    ax[2].plot(wps, sweep[i], "o-", ms=3, label=f"θ0={deg(th_):.0f}°")
ax[2].set(xlabel=r"$|\omega_{0,\perp}|$ [rad/s]", ylabel="max deviation [deg]", title="naive off-geodesic error"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig2_offgeodesic.png", dpi=130); plt.close()

# anisotropic gains break exactness even from rest
lawA = smd_law(np.diag([1., 4., 9.])*wn**2, np.diag([1., 2., 3.])*2*zeta*wn)
oA = sim2(q0, np.zeros(3), qd0, lawA, err="full", T=T)
EA, _, _ = lin2(th0*n0, np.zeros(3), lawA, T=T)
print_(f"  anisotropic gains, from rest: max | |r|-|x| | = "
       f"{deg(np.max(np.abs(oA['th'] - np.linalg.norm(EA, axis=1)))):.2f} deg")

# ================================================================== 3. LQR
print_("\n== 3. LQR")
qp, qv, rr = 10.0, 1.0, 0.1
kp, kd, P = lqr_double_integrator(qp, qv, rr)
try:
    from scipy.linalg import solve_continuous_are
    A = np.array([[0, 1], [0, 0.]]); B = np.array([[0], [-1.]])
    Ps = solve_continuous_are(A, B, np.diag([qp, qv]), np.array([[rr]]))
    print_("  closed-form P matches scipy CARE:", np.allclose(P, Ps), " kp, kd =", round(kp, 3), round(kd, 3))
except ImportError:
    pass
lawL = smd_law(kp, kd)
th0L = np.radians(120); q0L = qmul(qexp(-th0L*n0), qd0)
oL = sim2(q0L, np.zeros(3), qd0, lawL, err="full", T=6)
dtc = 1e-3
def cost(e, ed, u): return np.sum(qp*np.sum(e**2, 1) + qv*np.sum(ed**2, 1) + rr*np.sum(u**2, 1))*dtc
uL = oL["a"]                                           # = u (static target)
J_rot = cost(oL["r"], -oL["w"], uL)
J_opt = th0L**2*P[0, 0]
EL, EdL, UL = lin2(th0L*n0, np.zeros(3), lawL, T=6)
print_(f"  linear system, same Euler scheme: cost {cost(EL, EdL, UL):.4f}")
print_(f"  regulation 120 deg: rotational cost {J_rot:.4f}  vs  linear optimal z0'Pz0 = {J_opt:.4f}")
J_vec = None
oV = sim2(q0L, np.zeros(3), qd0, lawL, err="vec", T=6)
print_(f"  same gains with rho=1 error: cost (measured in true angle) {cost(oV['r'], -oV['w'], oV['a']):.4f}")

# tracking a moving target (tumbling reference)
wf = lambda t: np.array([0.8*np.sin(0.7*t), 0.6*np.cos(0.4*t), 0.5])
af = lambda t: np.array([0.56*np.cos(0.7*t), -0.24*np.sin(0.4*t), 0.0])
tgt = moving_target(wf, af)
q0T = qmul(qexp(-np.radians(100)*n0), qd0)
kpT, kdT, _ = lqr_double_integrator(4.0, 0.5, 0.1); lawT = smd_law(kpT, kdT)
Ttr = 8.0
trk = {m: sim2(q0T, np.zeros(3), qd0, lawT, err="full", T=Ttr, target=tgt, exact=(m == "exact"))
       for m in ("naive", "exact")}
trk["vec"] = sim2(q0T, np.zeros(3), qd0, lawT, err="vec", T=Ttr, target=tgt)
r0T = trk["naive"]["r"][0]
Etn, _, _ = lin2(r0T, wf(0) - np.zeros(3), lawT, T=Ttr)
Ete, _, _ = lin2(r0T, rdot(r0T, np.zeros(3), wf(0)), lawT, T=Ttr)
print_(f"  tracking: naive max | |r|-|x| | = {deg(np.max(np.abs(trk['naive']['th']-np.linalg.norm(Etn,axis=1)))):.2f} deg;"
       f" exact max |r-x| = {deg(np.max(np.linalg.norm(trk['exact']['r']-Ete,axis=1))):.4f} deg")
for m in trk: print_(f"    {m:5s} final error {deg(trk[m]['th'][-1]):.2e} deg")

tt = trk["naive"]["t"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(oL["t"], deg(th0L*np.ones_like(oL["t"])*0 + np.linalg.norm(lin2(th0L*n0, np.zeros(3), lawL, T=6)[0], axis=1)), "k", lw=3, alpha=.35, label="linear LQR")
ax[0].plot(oL["t"], deg(oL["th"]), label=r"$e=\theta n$")
ax[0].plot(oV["t"], deg(oV["th"]), "--", label=r"$e=\sin\frac{\theta}{2}n$")
ax[0].set(xlabel="t [s]", ylabel="angle [deg]", title="LQR regulation, 120°"); ax[0].legend(fontsize=8)
ax[1].plot(tt, deg(np.linalg.norm(Etn, axis=1)), "k", lw=3, alpha=.35, label="linear twin (naive)")
ax[1].plot(tt, deg(trk["naive"]["th"]), label="naive")
ax[1].plot(tt, deg(trk["exact"]["th"]), "C3--", label="exact")
ax[1].plot(tt, deg(trk["vec"]["th"]), "C2:", label=r"$\rho=1$")
ax[1].set(xlabel="t [s]", ylabel="tracking error [deg]", title="LQR tracking a tumbling target"); ax[1].legend(fontsize=8)
ax[2].semilogy(tt, deg(trk["naive"]["th"]) + 1e-9, label="naive"); ax[2].semilogy(tt, deg(trk["exact"]["th"]) + 1e-9, "C3--", label="exact")
ax[2].set(xlabel="t [s]", ylabel="deg (log)", title="tracking error, log scale"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig3_lqr.png", dpi=130); plt.close()

# ====================================================== 4. online minimum jerk
print_("\n== 4. online minimum-jerk (Hoff-Arbib, time-to-go feedback)")
lawJ = minjerk_law(); TJ = 2.0; th0J = np.radians(170)
q0J = qmul(qexp(-th0J*n0), qd0)
resJ = {v: sim3(q0J, np.zeros(3), np.zeros(3), qd0, lawJ, err=v, T=TJ + 0.3, Tf=TJ) for v in ("full", "half", "vec")}
tJ = resJ["full"]["t"]
thA, wA, aA = minjerk_profile(th0J, TJ, tJ)
EJ = lin3(th0J*n0, np.zeros(3), np.zeros(3), lawJ, T=TJ + 0.3, Tf=TJ)
for v, o in resJ.items():
    print_(f"  {v:5s}: max|angle-analytic| {deg(np.max(np.abs(o['th']-thA))):7.3f} deg ;"
           f" max|angle-linear sim| {deg(np.max(np.abs(o['th']-np.linalg.norm(EJ,axis=1)))):.2e} deg ;"
           f" final angle {deg(o['th'][-1]):.3e} deg")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(tJ, deg(thA), "k", lw=3, alpha=.35, label="analytic min-jerk")
for v, st in zip(resJ, ("-", "--", ":")): ax[0].plot(tJ, deg(resJ[v]["th"]), st, label=v)
ax[0].set(xlabel="t [s]", ylabel="angle [deg]", title="online min-jerk, 170°, T = 2 s"); ax[0].legend(fontsize=8)
ax[1].plot(tJ, wA, "k", lw=3, alpha=.35, label=r"analytic $\dot\theta$"); ax[1].plot(tJ, np.linalg.norm(resJ["full"]["w"], axis=1), label=r"$|\omega|$ (full)")
ax[1].plot(tJ, np.linalg.norm(resJ["half"]["w"], axis=1), "--", label=r"$|\omega|$ (half)")
ax[1].set(xlabel="t [s]", ylabel="rad/s", title="speed"); ax[1].legend(fontsize=8)
ax[2].plot(tJ, aA, "k", lw=3, alpha=.35, label=r"analytic $-\ddot\theta$"); ax[2].plot(tJ, resJ["full"]["a"]@n0, label="full")
ax[2].set(xlabel="t [s]", ylabel="rad/s²", title="angular acceleration along axis"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig4_minjerk.png", dpi=130); plt.close()

# goal switch mid-motion to an off-geodesic goal, new deadline
print_("  goal switch at t=0.8 s to a goal 90 deg off the original axis, new deadline 2.5 s")
qd_new = qmul(qexp(np.radians(90)*perp), qd0)
TS, Tnew = 0.8, 2.5
sw = sim3(q0J, np.zeros(3), np.zeros(3), qd0, lawJ, err="full", T=3.0, Tf=TJ, goal_switch=(TS, qd_new, Tnew))
ks = int(round(TS/1e-3))
# linear twin started just after the switch from (r, w_d-w, a_d-a) of the rotational state
rS = sw["r"][ks + 1]; Es = lin3(rS, -sw["w"][ks + 1], -sw["a"][ks + 1], lawJ, T=3.0 - TS - 1e-3, Tf=Tnew - TS - 1e-3)
thS = sw["th"][ks + 1:]
kf = int(round((Tnew - 0.02)/1e-3)) - 1
print_(f"    at deadline-20ms: angle {deg(sw['th'][kf]):.2e} deg, |w| {np.linalg.norm(sw['w'][kf]):.2e}, |a| {np.linalg.norm(sw['a'][kf]):.2e}; final angle {deg(sw['th'][-1]):.2e} deg")
print_(f"    max | |r| - |x_lin| | after switch = {deg(np.max(np.abs(thS - np.linalg.norm(Es, axis=1)))):.2f} deg")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(sw["t"], deg(sw["th"]), label="rotation (full log, naive rates)")
ax[0].plot(sw["t"][ks + 1:], deg(np.linalg.norm(Es, axis=1)), "k", lw=3, alpha=.35, label="linear twin from switch")
ax[0].axvline(TS, color="gray", ls=":"); ax[0].axvline(Tnew, color="gray", ls=":")
ax[0].set(xlabel="t [s]", ylabel="angle to current goal [deg]", title="online min-jerk with goal switch"); ax[0].legend(fontsize=8)
ax[1].plot(sw["t"], np.linalg.norm(sw["w"], axis=1), label=r"$|\omega|$"); ax[1].plot(sw["t"], np.linalg.norm(sw["a"], axis=1), label=r"$|\dot\omega|$")
ax[1].axvline(TS, color="gray", ls=":"); ax[1].set(xlabel="t [s]", title="rates stay continuous across the switch"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig5_minjerk_switch.png", dpi=130); plt.close()
print_("\nfigures written.")
