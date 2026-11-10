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
