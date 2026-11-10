"""Attitude + gyro-bias estimation: four EKF variants on identical simulated IMU streams.
  AQ   additive-quaternion EKF: state (q in R^4, b), q renormalized after each update
  MEKF multiplicative EKF, body-frame error q = qh (x) Exp(d)  (= left-invariant error)
  RIEKF right-invariant EKF as in the invariant-filtering literature:
        error q = Exp(x) (x) qh, innovation rotated to world frame z = Rh y - u, H = [u]x (constant)
  OURS error-state filter derived from the chart r = Log(q (x) qh^*):
        xi' = -J_r^{-1}(xi) Rh (db + n) linearized at xi = 0,  z = y - Rh^T u,  H = Rh^T [u]x
usage: python kf_compare.py <scenario> <n_runs> <out.npz>"""
import numpy as np, sys, time

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def qconj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qexp(r):
    th = np.sqrt(r @ r); return np.concatenate([[np.cos(th/2)], 0.5*np.sinc(th/(2*np.pi))*r])
def qlog(q):
    if q[0] < 0: q = -q
    th = 2*np.arctan2(np.sqrt(q[1:] @ q[1:]), q[0]); return q[1:]*(2/np.sinc(th/(2*np.pi)))
def qR(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def skew(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def Lmat(q):   # q (x) p = Lmat(q) p
    w, x, y, z = q
    return np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])
def Rmatq(p):  # q (x) p = Rmatq(p) q
    w, x, y, z = p
    return np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])

SCEN = {  # initial attitude error sigma per axis [deg], initial bias sigma [rad/s], rate scale
    "nominal":     dict(sig0=5,   sigb=0.02, wscale=1.0),
    "large_init":  dict(sig0=60,  sigb=0.02, wscale=1.0),
    "unknown_init":dict(sig0=None,sigb=0.02, wscale=1.0),   # uniformly random initial attitude
    "fast_motion": dict(sig0=20,  sigb=0.05, wscale=4.0),
}
dt, T, every = 0.01, 15.0, 10
sg, sb, sv = 0.01, 1e-3, 0.03
U = [np.array([0, 0, 1.]), np.array([0.6, 0, -0.8])]
NAMES = ("MEKFJ", "OURSJ")

def rand_quat(r):
    v = r.normal(size=4); return v/np.linalg.norm(v)

def run(seed, sc):
    r = np.random.default_rng(seed); N = int(T/dt)
    q = rand_quat(r); b = r.normal(size=3)*sc["sigb"]
    A1, A2 = r.uniform(.3, 1.2, 3)*sc["wscale"], r.uniform(.2, .8, 3)*sc["wscale"]
    f1, f2 = r.uniform(.1, .5, 3), r.uniform(.5, 1.5, 3)
    if sc["sig0"] is None:
        qh0 = rand_quat(r); s0 = np.pi/np.sqrt(3)      # sigma so that E|x|^2 = pi^2
    else:
        s0 = np.radians(sc["sig0"]); qh0 = qmul(qexp(-r.normal(size=3)*s0), q)
    Pa = np.eye(3)*s0**2; Pb = np.eye(3)*sc["sigb"]**2
    F = {}
    for n in NAMES:
        P = np.zeros((6, 6)); P[:3, :3] = Pa; P[3:, 3:] = Pb
        F[n] = dict(qh=qh0.copy(), bh=np.zeros(3), P=P)
    Xi = Lmat(qh0)[:, 1:]; Paq = np.zeros((7, 7)); Paq[:4, :4] = 0.25*Xi@Pa@Xi.T + 1e-9*np.eye(4); Paq[4:, 4:] = Pb
    pass
    E = {n: np.zeros(N) for n in NAMES}; NE = {n: np.zeros(N) for n in NAMES}; EB = {n: np.zeros(N) for n in NAMES}
    Qd = np.diag([sg**2]*3 + [sb**2*dt]*3); tim = {n: 0.0 for n in NAMES}
    for k in range(N):
        w = A1*np.sin(f1*k*dt) + A2*np.cos(f2*k*dt + 1)
        wm = w + b + r.normal(size=3)*sg
        q = qmul(q, qexp(w*dt)); q /= np.linalg.norm(q); b = b + r.normal(size=3)*sb*np.sqrt(dt)
        ys = [qR(q).T @ u + r.normal(size=3)*sv for u in U] if k % every == 0 else None
        for n in NAMES:
            t0 = time.perf_counter(); f = F[n]; qh, bh, P = f["qh"], f["bh"], f["P"]; wc = wm - bh
            if n == "AQ":
                dq = qexp(wc*dt); Fd = np.eye(7); Fd[:4, :4] = Rmatq(dq); Xi = Lmat(qh)[:, 1:]
                Fd[:4, 4:] = -0.5*dt*Xi; G = np.zeros((7, 6)); G[:4, :3] = -0.5*dt*Xi; G[4:, 3:] = np.eye(3)
                Qa = np.diag([sg**2/dt**2*dt**2]*3 + [sb**2*dt]*3)
                P = Fd@P@Fd.T + G@Qa@G.T; qh = qmul(qh, dq); qh /= np.linalg.norm(qh)
                if ys is not None:
                    z = np.concatenate([y - qR(qh).T@u for u, y in zip(U, ys)])
                    H = np.zeros((6, 7)); h = 1e-7
                    for j in range(4):
                        e = np.zeros(4); e[j] = h
                        H[:, j] = np.concatenate([(qR(qh+e).T@u - qR(qh-e).T@u)/(2*h) for u in U])
                    S = H@P@H.T + np.eye(6)*sv**2; K = np.linalg.solve(S, H@P).T
                    dx = K@z; P = (np.eye(7) - K@H)@P; P = 0.5*(P + P.T)
                    qh = qh + dx[:4]; qh /= np.linalg.norm(qh); bh = bh + dx[4:]
                Xi = Lmat(qh)[:, 1:]; Pth = 4*Xi.T@P[:4, :4]@Xi       # body-frame attitude covariance
                e = qlog(qmul(qconj(qh), q))
            else:
                Rh = qR(qh)
                if n in ("MEKF", "MEKFR", "MEKFJ"):
                    Fd = np.eye(6); Fd[:3, :3] = qR(qexp(-wc*dt)); Fd[:3, 3:] = -np.eye(3)*dt
                    G = np.zeros((6, 6)); G[:3, :3] = -np.eye(3)*dt; G[3:, 3:] = np.eye(3)
                else:
                    Fd = np.eye(6); Fd[:3, 3:] = -Rh*dt
                    G = np.zeros((6, 6)); G[:3, :3] = -Rh*dt; G[3:, 3:] = np.eye(3)
                P = Fd@P@Fd.T + G@Qd@G.T
                qh = qmul(qh, qexp(wc*dt)); qh /= np.linalg.norm(qh); Rh = qR(qh)
                if ys is not None:
                    H = np.zeros((6, 6)); z = np.zeros(6)
                    for i, (u, y) in enumerate(zip(U, ys)):
                        if n in ("MEKF", "MEKFR", "MEKFJ"):  z[3*i:3*i+3] = y - Rh.T@u;  H[3*i:3*i+3, :3] = skew(Rh.T@u)
                        if n in ("OURS", "OURSJ"):  z[3*i:3*i+3] = y - Rh.T@u;  H[3*i:3*i+3, :3] = Rh.T@skew(u)
                        if n == "RIEKF": z[3*i:3*i+3] = Rh@y - u;    H[3*i:3*i+3, :3] = skew(u)
                    S = H@P@H.T + np.eye(6)*sv**2; K = np.linalg.solve(S, H@P).T
                    dx = K@z; P = (np.eye(6) - K@H)@P; P = 0.5*(P + P.T)
                    qh = qmul(qh, qexp(dx[:3])) if n in ("MEKF", "MEKFR", "MEKFJ") else qmul(qexp(dx[:3]), qh)
                    qh /= np.linalg.norm(qh); bh = bh + dx[3:]
                    import sys as _s; _s.path.insert(0, "/mnt/user-data/outputs"); from rotdyn import Jl as _Jl
                    if n in ("MEKFJ", "OURSJ"):
                        C = np.eye(6); C[:3, :3] = _Jl(-dx[:3]) if n == "MEKFJ" else _Jl(dx[:3]); P = C@P@C.T
                    if n == "MEKFR":
                        C = np.eye(6); C[:3, :3] = qR(qexp(-dx[:3])); P = C@P@C.T
                Pth = P[:3, :3]
                e = qlog(qmul(qconj(qh), q)) if n in ("MEKF", "MEKFR", "MEKFJ") else qlog(qmul(q, qconj(qh)))
            f.update(qh=qh, bh=bh, P=P); tim[n] += time.perf_counter() - t0
            E[n][k] = np.linalg.norm(e); NE[n][k] = e@np.linalg.solve(Pth, e); EB[n][k] = np.linalg.norm(b - bh)
    return E, NE, EB, tim, F

if __name__ == "__main__":
    sc_name, M, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]; sc = SCEN[sc_name]
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    res = {f"{k}_{n}": [] for k in ("E", "NE", "EB") for n in NAMES}; dq = []; tt = {n: 0 for n in NAMES}
    for s in range(seed0, seed0 + M):
        E, NE, EB, tim, F = run(1000*list(SCEN).index(sc_name) + s, sc)
        for n in NAMES:
            res[f"E_{n}"].append(E[n]); res[f"NE_{n}"].append(NE[n]); res[f"EB_{n}"].append(EB[n]); tt[n] += tim[n]
        dq.append(np.linalg.norm(qlog(qmul(F["MEKFJ"]["qh"], qconj(F["OURSJ"]["qh"])))))
    np.savez(out, **{k: np.array(v) for k, v in res.items()}, dq=np.array(dq), **{f"time_{n}": tt[n] for n in NAMES})
    print(sc_name, "done", M, "runs; max final |RIEKF - OURS| =", max(dq), "rad")
