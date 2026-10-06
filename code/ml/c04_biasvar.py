"""Chapter 4: bias-variance decomposition, ridge, lasso, elastic net, early stopping, double descent."""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet, Lasso, Ridge, lasso_path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- bias-variance vs ridge lambda
f = lambda t: np.sin(2 * np.pi * t)
sigma, ntr, reps, deg = 0.3, 25, 200, 9
xte = np.linspace(0.05, 0.95, 50)
P = PolynomialFeatures(deg, include_bias=False)
lams = np.logspace(-8, 2, 26)
bias2, var, mse = [], [], []
Xte = P.fit_transform(xte[:, None])
train_sets = []
for r in range(reps):
    xtr = rng.uniform(0, 1, ntr)
    ytr = f(xtr) + rng.normal(0, sigma, ntr)
    train_sets.append((xtr, ytr))
for lam in lams:
    preds = np.empty((reps, len(xte)))
    for r, (xtr, ytr) in enumerate(train_sets):
        Xtr = P.transform(xtr[:, None])
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=lam).fit(sc.transform(Xtr), ytr)
        preds[r] = m.predict(sc.transform(Xte))
    mean_pred = preds.mean(0)
    b2 = np.mean((mean_pred - f(xte)) ** 2)
    v = np.mean(preds.var(0))
    # expected test error estimated with fresh noisy targets
    yte_noisy = f(xte)[None, :] + rng.normal(0, sigma, preds.shape)
    e = np.mean((preds - yte_noisy) ** 2)
    bias2.append(b2); var.append(v); mse.append(e)
bias2, var, mse = map(np.array, (bias2, var, mse))
out.dat("bv", {"lam": lams, "bias2": bias2, "var": var, "noise": np.full(len(lams), sigma ** 2),
               "total": bias2 + var + sigma ** 2, "mse": mse})
out.check("bias^2 + variance + noise = simulated test MSE", bias2 + var + sigma ** 2, mse, rtol=0.06, atol=0.01)
i_best = int(np.argmin(bias2 + var))
e_ = np.log10(lams[i_best])
out.val("bv_best_lam", "10^{%g}" % round(e_, 1))
for name, i in [("lo", 0), ("best", i_best), ("hi", len(lams) - 1)]:
    out.val(f"bv_b2_{name}", bias2[i], 3)
    out.val(f"bv_v_{name}", var[i], 3)
out.val("bv_lam_lo", "10^{-8}")
out.val("bv_lam_hi", "10^{2}")

# ---------------------------------------------------------------- ridge by hand (1-D, centred)
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) - 3
y = np.array([2.0, 4.0, 5.0, 4.0, 6.0]) - 4.2
for lam in [0, 1, 10, 40, 1000]:
    w = (x @ y) / (x @ x + lam)
    if lam > 0:
        out.check(f"1-D ridge lambda={lam} vs sklearn", w,
                  Ridge(alpha=lam, fit_intercept=False).fit(x[:, None], y).coef_[0])
    out.val(f"r1_{lam}", w, 4)

# ---------------------------------------------------------------- ridge closed form + SVD view
Xd, yd = load_diabetes(return_X_y=True)
Xs = StandardScaler().fit_transform(Xd)
yc = yd - yd.mean()
lam = 50.0
w_cf = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ yc)
w_sk = Ridge(alpha=lam).fit(Xs, yd).coef_
out.check("ridge closed form vs sklearn Ridge (diabetes, lambda=50)", w_cf, w_sk)
U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
w_svd = Vt.T @ ((s / (s ** 2 + lam)) * (U.T @ yc))
out.check("ridge via SVD shrinkage vs closed form", w_svd, w_cf)
dof = np.sum(s ** 2 / (s ** 2 + lam))
H = Xs @ np.linalg.solve(Xs.T @ Xs + lam * np.eye(10), Xs.T)
out.check("effective dof = trace of ridge hat matrix", dof, np.trace(H))
out.val("dof50", dof, 2)
out.val("shrink_top", s[0] ** 2 / (s[0] ** 2 + lam), 3)
out.val("shrink_bot", s[-1] ** 2 / (s[-1] ** 2 + lam), 3)
out.val("n_diab", Xs.shape[0]); out.val("p_diab", Xs.shape[1])

# ridge and lasso paths on diabetes
names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
ralphas = np.logspace(-2, 5, 60)
rcoef = np.array([Ridge(alpha=a).fit(Xs, yd).coef_ for a in ralphas])
out.dat("ridgepath", {"alpha": ralphas, **{n: rcoef[:, i] for i, n in enumerate(names)}})
lal, lcoef, _ = lasso_path(Xs, yc, alphas=80, eps=1e-3)
out.dat("lassopath", {"alpha": lal, **{n: lcoef[i] for i, n in enumerate(names)}})
# number of non-zeros at a few alphas
for a in [1.0, 5.0, 20.0]:
    m = Lasso(alpha=a).fit(Xs, yd)
    out.val(f"nnz_{int(a)}", int(np.sum(m.coef_ != 0)))
out.val("ridge_nnz_big", int(np.sum(np.abs(Ridge(alpha=1e4).fit(Xs, yd).coef_) > 0)))

# ---------------------------------------------------------------- lasso: soft-thresholding + coordinate descent
def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0)

# orthonormal design: lasso = soft-threshold OLS coefficients
ols = np.array([3.0, -1.2, 0.4, -0.1])
out.tex("soft_in", ", ".join(f"{v:g}" for v in ols))
out.tex("soft_out", ", ".join(f"{v + 0.0:g}" for v in soft(ols, 0.5)))
out.tex("ridge_out", ", ".join(f"{v:.3g}" for v in ols / (1 + 1.0)))
# verify on an orthonormal design with sklearn (sklearn objective: 1/(2n)||y-Xw||^2 + a||w||_1)
n = 100
Qo, _ = np.linalg.qr(rng.normal(size=(n, 4)))
yq = Qo @ ols + 0.0
alpha_sk = 0.5 / n  # gives threshold 0.5 on X^T y for unit-norm columns
m = Lasso(alpha=alpha_sk, fit_intercept=False, tol=1e-12, max_iter=100000).fit(Qo, yq)
out.check("lasso on orthonormal X = soft-threshold(OLS, lambda)", m.coef_, soft(ols, 0.5), atol=1e-6)

# [[cd]]
def lasso_cd(X, y, lam, iters=500):
    """min 1/(2n)||y - Xw||^2 + lam ||w||_1 by cyclic coordinate descent."""
    n, d = X.shape
    w = np.zeros(d)
    col_sq = (X ** 2).sum(0) / n
    r = y - X @ w
    for _ in range(iters):
        for j in range(d):
            r += X[:, j] * w[j]                      # remove j's contribution
            rho = X[:, j] @ r / n
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / col_sq[j]
            r -= X[:, j] * w[j]
    return w
# [[/cd]]
w_cd = lasso_cd(Xs, yc, 1.0)
w_ref = Lasso(alpha=1.0, tol=1e-10, max_iter=100000).fit(Xs, yd).coef_
out.check("our coordinate-descent lasso vs sklearn Lasso (diabetes, alpha=1)", w_cd, w_ref, atol=1e-4)

# ---------------------------------------------------------------- elastic net grouping effect
rg = np.random.default_rng(0)
n = 200
z = rg.normal(size=n)
x1 = z + 0.01 * rg.normal(size=n)
x2 = z + 0.01 * rg.normal(size=n)
x3 = rg.normal(size=n)
Xg = np.column_stack([x1, x2, x3])
yg = 2 * z + x3 + 0.5 * rg.normal(size=n)
la = Lasso(alpha=0.1).fit(Xg, yg).coef_
en = ElasticNet(alpha=0.1, l1_ratio=0.3).fit(Xg, yg).coef_
rd = Ridge(alpha=10).fit(Xg, yg).coef_
out.tex("g_lasso", ", ".join(f"{v:.2f}" for v in la))
out.tex("g_en", ", ".join(f"{v:.2f}" for v in en))
out.tex("g_ridge", ", ".join(f"{v:.2f}" for v in rd))

# ---------------------------------------------------------------- early stopping
n, d = 60, 50
Xe = rng.normal(size=(n, d))
we = np.zeros(d); we[:5] = [3, -2, 2, 1, -1]
ye = Xe @ we + rng.normal(0, 2.0, n)
Xv = rng.normal(size=(2000, d)); yv = Xv @ we + rng.normal(0, 2.0, 2000)
w = np.zeros(d); eta = 0.002
tr_c, va_c, its = [], [], []
for t in range(3001):
    if t % 10 == 0:
        its.append(t)
        tr_c.append(np.mean((Xe @ w - ye) ** 2))
        va_c.append(np.mean((Xv @ w - yv) ** 2))
    w -= eta * 2 / n * Xe.T @ (Xe @ w - ye)
out.dat("early", {"it": its, "train": tr_c, "val": va_c})
k = int(np.argmin(va_c))
out.val("es_best_it", its[k]); out.val("es_best_val", va_c[k], 2)
out.val("es_final_val", va_c[-1], 2); out.val("es_final_tr", tr_c[-1], 2)

# ---------------------------------------------------------------- double descent (min-norm least squares)
n, D, reps = 40, 200, 100
beta = rng.normal(size=D) / np.sqrt(D)
ps = np.arange(5, 201, 5)
ps = np.unique(np.concatenate([ps, [36, 38, 39, 40, 41, 42, 44]]))
errs = np.zeros(len(ps))
for _ in range(reps):
    Xa = rng.normal(size=(n, D)); ya = Xa @ beta + 0.1 * rng.normal(size=n)
    Xt = rng.normal(size=(500, D)); yt = Xt @ beta + 0.1 * rng.normal(size=500)
    for i, p in enumerate(ps):
        wmn = np.linalg.pinv(Xa[:, :p]) @ ya
        errs[i] += np.mean((Xt[:, :p] @ wmn - yt) ** 2)
errs /= reps
out.dat("dd", {"p": ps, "err": np.minimum(errs, 20)})
out.val("dd_peak_p", int(ps[np.argmax(errs)]))
out.val("dd_n", n)
out.val("dd_err_p200", errs[-1], 2)
out.val("dd_err_p20", errs[list(ps).index(20)], 2)
out.val("dd_min_under", errs[ps < n].min(), 2)

# ---------------------------------------------------------------- geometry for the l1/l2 figure
# RSS contours: ellipses centred at c with semi-axes (0.5 r, 1.0 r) rotated by 50 degrees.
# Find the level r at which each contour first touches the l1 diamond / l2 disc of radius 1.3.
cen = np.array([1.0, 2.0]); th = np.deg2rad(50)
Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
def level(p):
    q = Rot.T @ (p - cen)
    return np.sqrt((q[0] / 0.5) ** 2 + (q[1] / 1.0) ** 2)
ang = np.linspace(0, 2 * np.pi, 20001)
disc = 1.3 * np.column_stack([np.cos(ang), np.sin(ang)])
diam = 1.3 * np.column_stack([np.cos(ang), np.sin(ang)]) / (np.abs(np.cos(ang)) + np.abs(np.sin(ang)))[:, None]
for nm, pts in [("l1", diam), ("l2", disc)]:
    lv = np.array([level(p) for p in pts])
    k = int(np.argmin(lv))
    out.val(f"geo_{nm}_r", lv[k], 4)
    out.val(f"geo_{nm}_x", pts[k, 0], 3)
    out.val(f"geo_{nm}_y", pts[k, 1], 3)
