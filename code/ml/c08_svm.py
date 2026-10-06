"""Chapter 8: SVM — hard margin by hand, dual QP vs sklearn, soft margin and C, kernels, SVR."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.svm import SVC, SVR
from common import sci, setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- hard margin by hand
X = np.array([[0, 0], [1, 0], [0, 1], [2, 2], [3, 3], [2, 3]], float)
y = np.array([-1, -1, -1, 1, 1, 1], float)
svc = SVC(kernel="linear", C=1e8, tol=1e-10).fit(X, y)
w_sk, b_sk = svc.coef_[0], svc.intercept_[0]
out.check("hard-margin w vs hand (2/3, 2/3)", w_sk, [2 / 3, 2 / 3], atol=1e-5)
out.check("hard-margin b vs hand -5/3", b_sk, -5 / 3, atol=1e-5)
out.val("margin", 2 / np.linalg.norm(w_sk), 4)
out.check("margin width 2/||w|| = 3/sqrt(2)", 2 / np.linalg.norm(w_sk), 3 / np.sqrt(2), atol=1e-5)
sv = sorted(map(tuple, svc.support_vectors_.astype(int).tolist()))
out.tex("svs", ", ".join(f"({a},{b})" for a, b in sv))
# dual coefficients alpha_i (sklearn stores y_i alpha_i)
alpha_sk = np.zeros(len(X)); alpha_sk[svc.support_] = np.abs(svc.dual_coef_[0])
out.check("dual alphas vs hand (2/9, 2/9, 4/9 on the SVs)", alpha_sk, [0, 2 / 9, 2 / 9, 4 / 9, 0, 0], atol=1e-5)

# [[dual]]
def svm_dual(K, y, C=np.inf):
    """max sum(a) - 1/2 a^T (yy^T * K) a  s.t.  0 <= a_i <= C, sum a_i y_i = 0."""
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K
    obj = lambda a: 0.5 * a @ Q @ a - a.sum()
    jac = lambda a: Q @ a - 1
    bounds = [(0, None if np.isinf(C) else C)] * n
    res = minimize(obj, np.zeros(n), jac=jac, bounds=bounds, method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x
# [[/dual]]
a = svm_dual(X @ X.T, y)
out.check("our dual QP (SLSQP) alphas vs sklearn", a, alpha_sk, atol=1e-5)
w_dual = (a * y) @ X
out.check("w = sum a_i y_i x_i vs sklearn", w_dual, w_sk, atol=1e-5)

# ---------------------------------------------------------------- soft margin: effect of C
Xb, yb = make_blobs(n_samples=80, centers=[[0, 0], [2.2, 2.2]], cluster_std=1.0, random_state=4)
yb = 2 * yb - 1
Xt, yt = make_blobs(n_samples=2000, centers=[[0, 0], [2.2, 2.2]], cluster_std=1.0, random_state=5)
yt = 2 * yt - 1
rows = []
for C in [0.01, 0.1, 1, 10, 100]:
    m = SVC(kernel="linear", C=C).fit(Xb, yb)
    wv = m.coef_[0]
    rows.append(f"{C:g} & {2 / np.linalg.norm(wv):.2f} & {len(m.support_)} & {m.score(Xb, yb):.3f} & {m.score(Xt, yt):.3f} \\\\")
    if C in (0.01, 100):
        tag = "lo" if C == 0.01 else "hi"
        xs = np.linspace(-3, 5, 2)
        bb = m.intercept_[0]
        cols = {"x": xs}
        for k, off in [("mid", 0), ("up", 1), ("dn", -1)]:
            cols[k] = (off - bb - wv[0] * xs) / wv[1]
        out.dat(f"soft_{tag}", cols)
        out.dat(f"softsv_{tag}", {"x": m.support_vectors_[:, 0], "y": m.support_vectors_[:, 1]})
out.tex("C_rows", "\n".join(rows))
out.dat("blob_p", {"x": Xb[yb == 1, 0], "y": Xb[yb == 1, 1]})
out.dat("blob_n", {"x": Xb[yb == -1, 0], "y": Xb[yb == -1, 1]})

# soft-margin primal solved as a QP vs sklearn (linear kernel, C = 1)
n = len(yb); C = 1.0
def primal_obj(z):
    wv, bv, xi = z[:2], z[2], z[3:]
    return 0.5 * wv @ wv + C * xi.sum()
cons = [{"type": "ineq", "fun": lambda z: yb * (Xb @ z[:2] + z[2]) - 1 + z[3:]}]
res = minimize(primal_obj, np.zeros(3 + n), constraints=cons, bounds=[(None, None)] * 3 + [(0, None)] * n,
               method="SLSQP", options={"ftol": 1e-12, "maxiter": 2000})
m1 = SVC(kernel="linear", C=1.0, tol=1e-8).fit(Xb, yb)
out.check("soft-margin primal QP (w, b) vs sklearn SVC(C=1)", res.x[:3],
          np.r_[m1.coef_[0], m1.intercept_], atol=2e-3)
xi = res.x[3:]
hinge = np.maximum(0, 1 - yb * (Xb @ res.x[:2] + res.x[2]))
out.check("optimal slacks equal hinge losses", xi, hinge, atol=2e-3)
out.val("n_margin_viol", int(np.sum(hinge > 1e-3)))
out.val("n_misclass", int(np.sum(hinge > 1)))

# ---------------------------------------------------------------- kernel trick numerics
x1, z1 = np.array([1.0, 2.0]), np.array([3.0, 1.0])
def phi(v):
    return np.array([1, np.sqrt(2) * v[0], np.sqrt(2) * v[1], v[0] ** 2, v[1] ** 2, np.sqrt(2) * v[0] * v[1]])
kval = (x1 @ z1 + 1) ** 2
out.check("(x.z+1)^2 = phi(x).phi(z) (6-dim map)", kval, phi(x1) @ phi(z1))
out.check("(x.z+1)^2 vs sklearn polynomial_kernel", kval, polynomial_kernel([x1], [z1], degree=2, gamma=1, coef0=1)[0, 0])
out.val("poly_dot", x1 @ z1, 0); out.val("poly_k", kval, 0)
out.tex("phi_x", ", ".join(f"{v:.3g}" for v in phi(x1)))
gam = 0.5
rbf = np.exp(-gam * np.sum((x1 - z1) ** 2))
out.check("RBF kernel by formula vs sklearn", rbf, rbf_kernel([x1], [z1], gamma=gam)[0, 0])
out.val("rbf_sq", np.sum((x1 - z1) ** 2), 0); out.val("rbf", rbf, 4)
# number of monomial features of degree <= p in d dims vs kernel cost
out.val("poly_dim_100_3", int(np.round(np.prod(range(101, 104)) / 6)))  # C(103,3)
# PSD check: RBF Gram matrix vs a "kernel" that is not PSD
Z = rng.normal(size=(30, 2))
ev_rbf = np.linalg.eigvalsh(rbf_kernel(Z, gamma=1.0)).min()
Dm = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
ev_bad = np.linalg.eigvalsh(-Dm).min()
out.val("ev_rbf", ev_rbf, 4 if ev_rbf > 1e-4 else 6)
out.val("ev_rbf_sci", sci(ev_rbf))
out.val("ev_bad", ev_bad, 2)

# ---------------------------------------------------------------- RBF SVM on moons: effect of gamma, C
Xm, ym = make_moons(n_samples=200, noise=0.25, random_state=0)
Xmt, ymt = make_moons(n_samples=2000, noise=0.25, random_state=1)
gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 220), np.linspace(-1.2, 1.7, 160))
G = np.column_stack([gx.ravel(), gy.ravel()])
grows = []
for gm in [0.1, 1, 10, 100]:
    m = SVC(kernel="rbf", gamma=gm, C=1).fit(Xm, ym)
    grows.append(f"{gm:g} & {len(m.support_)} & {m.score(Xm, ym):.3f} & {m.score(Xmt, ymt):.3f} \\\\")
    if gm in (1, 100):
        Zg = m.decision_function(G).reshape(gx.shape)
        cs = plt.contour(gx, gy, Zg, levels=[0])
        segs = cs.allsegs[0]
        # concatenate segments separated by blank lines (pgfplots treats NaN rows as jumps)
        xsall, ysall = [], []
        for sgm in segs:
            if len(sgm) < 5:
                continue
            xsall += list(sgm[::2, 0]) + [np.nan]; ysall += list(sgm[::2, 1]) + [np.nan]
        out.dat(f"rbf_g{gm}", {"x": np.array(xsall), "y": np.array(ysall)})
        plt.close("all")
out.tex("gamma_rows", "\n".join(grows))
out.dat("moon0", {"x": Xm[ym == 0, 0], "y": Xm[ym == 0, 1]})
out.dat("moon1", {"x": Xm[ym == 1, 0], "y": Xm[ym == 1, 1]})
# linear vs RBF on circles
Xc, yc = make_circles(n_samples=300, noise=0.08, factor=0.4, random_state=0)
out.val("circ_lin", SVC(kernel="linear").fit(Xc, yc).score(Xc, yc), 3)
out.val("circ_rbf", SVC(kernel="rbf").fit(Xc, yc).score(Xc, yc), 3)
Xc2 = np.column_stack([Xc, (Xc ** 2).sum(1)])
out.val("circ_lift", SVC(kernel="linear").fit(Xc2, yc).score(Xc2, yc), 3)

# ---------------------------------------------------------------- SVR: epsilon tube
xs = np.sort(rng.uniform(0, 4, 60)); ys = np.sin(xs) + rng.normal(0, 0.15, 60)
srows = []
for eps in [0.01, 0.1, 0.3, 0.6]:
    m = SVR(kernel="rbf", C=10, epsilon=eps, gamma=1.0).fit(xs[:, None], ys)
    srows.append(f"{eps:g} & {len(m.support_)} & {np.mean((m.predict(xs[:, None]) - ys) ** 2):.4f} \\\\")
    if eps == 0.3:
        grid = np.linspace(0, 4, 100)
        pr = m.predict(grid[:, None])
        out.dat("svr", {"x": grid, "f": pr, "up": pr + eps, "dn": pr - eps})
        out.dat("svr_sv", {"x": xs[m.support_], "y": ys[m.support_]})
out.dat("svr_pts", {"x": xs, "y": ys})
out.tex("svr_rows", "\n".join(srows))
