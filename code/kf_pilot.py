"""Pilot: attitude + gyro-bias error-state EKF, gyro + two vector sensors (gravity, magnetic field).
  B  : body-frame error  q = qh (x) Exp(d)      (classic MEKF), reset without covariance correction
  W  : world-frame error q = Exp(x) (x) qh      (our chart, right-invariant), reset without correction
  WJ : W + reset covariance correction P <- J_l(x+) P J_l(x+)^T
Monte Carlo over random trajectories and large initial errors. Metrics: attitude RMSE, NEES (ideal = 3)."""
import numpy as np, sys
from rotdyn import qmul, qconj, qexp, qlog, qnormalize, Rmat, Jl, skew
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
dt, T, meas_every = 0.01, 15.0, 10                      # 100 Hz gyro, 10 Hz vector sensors
sg, sb, sv = 0.01, 1e-3, 0.03                           # gyro noise [rad/s], bias RW [rad/s/sqrt(s)], vector noise
sig0 = np.radians(float(sys.argv[1]) if len(sys.argv) > 1 else 60)
U = [np.array([0, 0, 1.]), np.array([0.6, 0, -0.8])]   # gravity "up", magnetic field direction (world)
N, MC = int(T/dt), int(sys.argv[3]) if len(sys.argv) > 3 else 30
R_of = lambda q: Rmat(qlog(q))

def run(seed):
    r = np.random.default_rng(seed)
    q = qexp(r.normal(size=3)); b = r.normal(size=3)*0.02
    A1, A2, f1, f2 = r.uniform(0.3, 1.2, 3), r.uniform(0.2, 0.8, 3), r.uniform(0.1, 0.5, 3), r.uniform(0.5, 1.5, 3)
    wfun = lambda t: A1*np.sin(f1*t) + A2*np.cos(f2*t + 1)
    x0 = r.normal(size=3)*sig0
    P0 = np.diag([sig0**2]*3 + [0.02**2]*3)
    # same initial estimate for all filters: qh = Exp(-x0) q  (world error x0)
    qh0 = qmul(qexp(-x0), q)
    F = {k: dict(qh=qh0.copy(), bh=np.zeros(3), P=P0.copy()) for k in ("B", "W", "WJ")}
    # body-frame initial covariance: d = R^T x  -> rotate P
    Rh = R_of(qh0); F["B"]["P"][:3, :3] = Rh.T @ P0[:3, :3] @ Rh
    err = {k: np.zeros(N) for k in F}; nees = {k: np.zeros(N) for k in F}
    for k in range(N):
        t = k*dt; w = wfun(t)
        wm = w + b + r.normal(size=3)*sg
        q = qnormalize(qmul(q, qexp(w*dt))); b = b + r.normal(size=3)*sb*np.sqrt(dt)
        ys = [R_of(q).T @ u + r.normal(size=3)*sv for u in U] if k % meas_every == 0 else None
        for name, f in F.items():
            qh, bh, P = f["qh"], f["bh"], f["P"]
            Rh = R_of(qh); wc = wm - bh
            # --- propagate
            if name == "B":
                Fd = np.eye(6); Fd[:3, :3] = Rmat(-wc*dt); Fd[:3, 3:] = -np.eye(3)*dt
                G = np.zeros((6, 6)); G[:3, :3] = -np.eye(3)*dt; G[3:, 3:] = np.eye(3)
            else:
                Fd = np.eye(6); Fd[:3, 3:] = -Rh*dt
                G = np.zeros((6, 6)); G[:3, :3] = -Rh*dt; G[3:, 3:] = np.eye(3)
            Qd = np.diag([sg**2]*3 + [sb**2*dt]*3)
            P = Fd @ P @ Fd.T + G @ Qd @ G.T
            qh = qnormalize(qmul(qh, qexp(wc*dt))); Rh = R_of(qh)
            # --- update
            if ys is not None:
                H = np.zeros((6, 6)); z = np.zeros(6)
                for i, (u, y) in enumerate(zip(U, ys)):
                    yh = Rh.T @ u; z[3*i:3*i+3] = y - yh
                    H[3*i:3*i+3, :3] = skew(yh) if name == "B" else Rh.T @ skew(u)
                S = H @ P @ H.T + np.eye(6)*sv**2; K = P @ H.T @ np.linalg.inv(S)
                dx = K @ z; P = (np.eye(6) - K @ H) @ P; P = 0.5*(P + P.T)
                a = dx[:3]
                qh = qmul(qh, qexp(a)) if name == "B" else qmul(qexp(a), qh)
                bh = bh + dx[3:]
                if name == "WJ":
                    Tm = np.eye(6); Tm[:3, :3] = Jl(a); P = Tm @ P @ Tm.T
            f.update(qh=qh, bh=bh, P=P)
            e = qlog(qmul(q, qconj(qh))) if name != "B" else qlog(qmul(qconj(qh), q))
            err[name][k] = np.linalg.norm(e)
            nees[name][k] = e @ np.linalg.solve(P[:3, :3], e)
    return err, nees

E = {k: [] for k in ("B", "W", "WJ")}; NE = {k: [] for k in E}
for s in range(MC):
    e, n = run(1000 + s)
    for k in E: E[k].append(e[k]); NE[k].append(n[k])
t = np.arange(N)*dt
print(f"initial error sigma = {np.degrees(sig0):.0f} deg per axis, {MC} runs")
for k in E:
    e = np.degrees(np.array(E[k])); n = np.array(NE[k])
    rm = lambda a, b: np.sqrt(np.mean(e[:, (t >= a) & (t < b)]**2))
    print(f"  {k:3s}: RMSE 0-1 s {rm(0,1):6.2f} | 1-5 s {rm(1,5):6.3f} | 8-15 s {rm(8,15):6.3f} deg ;"
          f"  mean NEES 0-1 s {n[:, t<1].mean():8.2f} | 8-15 s {n[:, t>=8].mean():5.2f} ; diverged(>10deg at end) {(e[:,-1]>10).sum()}")
