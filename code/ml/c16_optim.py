"""Chapter 16: GD on quadratics, condition number, momentum, Newton, quasi-Newton, SGD noise."""
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

def quad_iters(kappa, method, tol=1e-6, max_it=100000):
    A = np.diag([1.0, kappa]); x = np.array([1.0, 1.0]); v = np.zeros(2)
    mu, L = 1.0, kappa
    if method == "gd_1L":
        eta = 1 / L
    elif method == "gd_opt":
        eta = 2 / (mu + L)
    else:  # heavy-ball with optimal constants
        eta = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
        beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
    path = [x.copy()]
    for it in range(1, max_it):
        g = A @ x
        if method == "heavy":
            v = beta * v - eta * g; x = x + v
        else:
            x = x - eta * g
        path.append(x.copy())
        if np.linalg.norm(x) < tol:
            return it, np.array(path)
    return max_it, np.array(path)

rows = []
for kappa in (10, 100, 1000):
    i1, _ = quad_iters(kappa, "gd_1L"); i2, _ = quad_iters(kappa, "gd_opt"); i3, _ = quad_iters(kappa, "heavy")
    rows.append(f"{kappa} & {i1} & {i2} & {i3} & {(kappa - 1) / (kappa + 1):.4f} & {(np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1):.4f} \\\\")
out.tex("rate_rows", "\n".join(rows))
# rate check: GD with eta=2/(mu+L) contracts the error by exactly (k-1)/(k+1) per step on the worst eigen-direction
_, p = quad_iters(10, "gd_opt")
ratio = np.abs(p[6, 0] / p[5, 0])
out.check("GD error ratio per step = (kappa-1)/(kappa+1)", ratio, 9 / 11)
# paths for the figure (kappa = 10), start (1, 0.5)
A10 = np.diag([1.0, 10.0])
def run(step, beta, n=15):
    x = np.array([1.0, 0.5]); v = np.zeros(2); P = [x.copy()]
    for _ in range(n):
        v = beta * v - step * A10 @ x; x = x + v; P.append(x.copy())
    return np.array(P)
pg = run(0.18, 0.0)                                            # GD with eta = 2/(mu+L)
ph = run(4 / (np.sqrt(10) + 1) ** 2, ((np.sqrt(10) - 1) / (np.sqrt(10) + 1)) ** 2)
out.dat("path_gd", {"x": pg[:, 0], "y": pg[:, 1]})
out.dat("path_hb", {"x": ph[:, 0], "y": ph[:, 1]})
out.val("gd15", np.linalg.norm(pg[-1]), 4); out.val("hb15", np.linalg.norm(ph[-1]), 4)
# divergence beyond 2/L
A = np.diag([1.0, 10.0]); x = np.array([1.0, 1.0])
for _ in range(20):
    x = x - 0.21 * A @ x
out.val("div_norm", np.linalg.norm(x), 1)
out.val("eta_max", 2 / 10, 2)

# ---------------------------------------------------------------- Newton on a quadratic: one step
b = np.array([1.0, 2.0])
x0 = np.array([5.0, -3.0])
x1 = x0 - np.linalg.solve(A, A @ x0 - b)
out.check("one Newton step lands on the minimiser A^{-1} b", x1, np.linalg.solve(A, b))

# ---------------------------------------------------------------- Rosenbrock: GD vs BFGS vs Newton-CG
x = np.array([-1.2, 1.0]); it_gd = 0
while np.linalg.norm(rosen_der(x)) > 1e-5 and it_gd < 200000:
    x = x - 1e-3 * rosen_der(x); it_gd += 1
out.val("ros_gd", it_gd)
for meth in ("BFGS", "L-BFGS-B", "Newton-CG"):
    kw = {"jac": rosen_der}
    if meth == "Newton-CG":
        from scipy.optimize import rosen_hess
        kw["hess"] = rosen_hess
    r = minimize(rosen, [-1.2, 1.0], method=meth, options={} if meth == "Newton-CG" else {"gtol": 1e-5}, **kw)
    out.val("ros_" + meth.replace("-", ""), r.nit)
    out.check(f"{meth} reaches the minimum (1, 1)", r.x, [1.0, 1.0], atol=1e-3)

# ---------------------------------------------------------------- SGD: constant vs decaying step
n, d = 2000, 5
X = rng.normal(size=(n, d)); w_true = rng.normal(size=d); y = X @ w_true + rng.normal(0, 1.0, n)
w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
f_ls = np.mean((X @ w_ls - y) ** 2)
curves = {}
for name, sched in [("const", lambda t: 0.02), ("decay", lambda t: 0.05 / (1 + t / 500))]:
    w = np.zeros(d); r2 = np.random.default_rng(1); gaps = []
    for t in range(20000):
        i = r2.integers(n)
        w -= sched(t) * 2 * (X[i] @ w - y[i]) * X[i]
        if t % 200 == 0:
            gaps.append(np.mean((X @ w - y) ** 2) - f_ls)
    curves[name] = np.array(gaps)
out.dat("sgd", {"t": np.arange(0, 20000, 200), "const": curves["const"], "decay": curves["decay"]})
out.val("sgd_const_end", np.mean(curves["const"][-10:]), 4); out.val("sgd_decay_end", np.mean(curves["decay"][-10:]), 4)
