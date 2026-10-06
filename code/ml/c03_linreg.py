"""Chapter 3: linear regression — hand fit, normal equations, GD trace, R^2, VIF, polynomials."""
from math import comb

import numpy as np
from scipy import stats
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- five points by hand
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.0, 5.0, 4.0, 6.0])
n = len(x)
Sx, Sy, Sxy, Sxx = x.sum(), y.sum(), (x * y).sum(), (x * x).sum()
b1 = (n * Sxy - Sx * Sy) / (n * Sxx - Sx ** 2)
b0 = (Sy - b1 * Sx) / n
ref = LinearRegression().fit(x[:, None], y)
out.check("5-point slope vs sklearn", b1, ref.coef_[0])
out.check("5-point intercept vs sklearn", b0, ref.intercept_)
out.check("5-point fit vs numpy.polyfit", [b1, b0], np.polyfit(x, y, 1))
for k, v in dict(Sx=Sx, Sy=Sy, Sxy=Sxy, Sxx=Sxx).items():
    out.val(k, v, 0)
out.val("xbar", x.mean(), 0)
out.val("ybar", y.mean(), 1)
out.val("cxy", ((x - x.mean()) * (y - y.mean())).sum(), 0)
out.val("cxx", ((x - x.mean()) ** 2).sum(), 0)
out.val("b1", b1, 2)
out.val("b0", b0, 2)
yhat = b0 + b1 * x
res = y - yhat
out.tex("yhat", " & ".join(f"{v:.1f}" for v in yhat))
out.tex("res", " & ".join(f"{v:+.1f}" for v in res))
out.val("rss", (res ** 2).sum(), 2)
out.val("tss", ((y - y.mean()) ** 2).sum(), 1)
R2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
out.val("R2", R2, 4)
out.check("R^2 vs sklearn r2_score", R2, r2_score(y, yhat))
r = np.corrcoef(x, y)[0, 1]
out.check("R^2 = r^2 (simple regression)", R2, r ** 2)
out.val("r", r, 4)
out.val("pred6", b0 + b1 * 6, 2)
out.val("sum_res", res.sum(), 3)
out.val("sum_xres", (x * res).sum(), 3)
# residual standard error and slope SE / t
s2 = (res ** 2).sum() / (n - 2)
se_b1 = np.sqrt(s2 / ((x - x.mean()) ** 2).sum())
lr = stats.linregress(x, y)
out.check("slope standard error vs scipy.linregress", se_b1, lr.stderr)
out.val("se_b1", se_b1, 4)
out.val("t_b1", b1 / se_b1, 3)
out.val("p_b1", lr.pvalue, 4)
out.dat("pts", {"x": x, "y": y, "yhat": yhat})

# ---------------------------------------------------------------- normal equations, multivariate
N, d = 200, 3
X = rng.normal(size=(N, d))
w_true = np.array([2.0, -1.0, 0.5])
yv = 3.0 + X @ w_true + rng.normal(0, 0.5, N)
Xb = np.column_stack([np.ones(N), X])
w_ne = np.linalg.solve(Xb.T @ Xb, Xb.T @ yv)
Qm, Rm = np.linalg.qr(Xb)
w_qr = np.linalg.solve(Rm, Qm.T @ yv)
w_ls = np.linalg.lstsq(Xb, yv, rcond=None)[0]
sk = LinearRegression().fit(X, yv)
w_sk = np.concatenate([[sk.intercept_], sk.coef_])
out.check("normal equations vs sklearn", w_ne, w_sk)
out.check("QR solution vs sklearn", w_qr, w_sk)
out.check("lstsq vs sklearn", w_ls, w_sk)
out.tex("w_ne", ", ".join(f"{v:.3f}" for v in w_ne))
# conditioning: kappa(X^T X) = kappa(X)^2
Xc = np.column_stack([np.ones(50), np.linspace(1000, 1001, 50)])
out.val("kX", "%.1f\\times 10^{%d}" % (np.linalg.cond(Xc) / 10 ** int(np.log10(np.linalg.cond(Xc))), int(np.log10(np.linalg.cond(Xc)))))
out.val("kXtX", "%.1f\\times 10^{%d}" % (np.linalg.cond(Xc.T @ Xc) / 10 ** int(np.log10(np.linalg.cond(Xc.T @ Xc))), int(np.log10(np.linalg.cond(Xc.T @ Xc)))))
out.check("cond(X^T X) = cond(X)^2", np.log10(np.linalg.cond(Xc.T @ Xc)), 2 * np.log10(np.linalg.cond(Xc)), atol=1e-2)

# ---------------------------------------------------------------- gradient descent trace on the 5 points
w0, w1, eta = 0.0, 0.0, 0.05
rows, losses = [], []
for t in range(0, 501):
    pred = w0 + w1 * x
    loss = np.mean((pred - y) ** 2)
    g0 = 2 * np.mean(pred - y)
    g1 = 2 * np.mean((pred - y) * x)
    if t < 4:
        rows.append(f"{t} & {w0:.4f} & {w1:.4f} & {loss:.4f} & {g0:.4f} & {g1:.4f} \\\\")
    losses.append(loss)
    w0, w1 = w0 - eta * g0, w1 - eta * g1
out.tex("gd_rows", "\n".join(rows))
out.check("GD after 500 steps vs closed form", [w0, w1], [b0, b1], atol=2e-3)
out.val("gd_w0", w0, 4)
out.val("gd_w1", w1, 4)
it = np.arange(len(losses))
sel = np.unique(np.concatenate([np.arange(0, 50), np.arange(50, 501, 5)]))
out.dat("gd", {"t": it[sel], "loss": np.array(losses)[sel]})
out.val("loss_min", np.mean(res ** 2), 4)
# divergence threshold: eta < 2/lambda_max(2/n X^T X)
Hmat = 2 / n * np.column_stack([np.ones(n), x]).T @ np.column_stack([np.ones(n), x])
lmax = np.linalg.eigvalsh(Hmat).max()
out.val("eta_max", 2 / lmax, 4)

# ---------------------------------------------------------------- adjusted R^2 with junk features
N2, reps = 40, 300
r2s = np.zeros((reps, 21))
adjs = np.zeros((reps, 21))
for rep in range(reps):
    x1 = rng.normal(size=N2)
    y2 = 1 + 2 * x1 + rng.normal(0, 1.5, N2)
    Xj = np.column_stack([x1, rng.normal(size=(N2, 20))])
    for k in range(0, 21):
        p = k + 1
        m = LinearRegression().fit(Xj[:, :p], y2)
        r2 = m.score(Xj[:, :p], y2)
        r2s[rep, k] = r2
        adjs[rep, k] = 1 - (1 - r2) * (N2 - 1) / (N2 - p - 1)
dat_k = np.arange(21)
dat_r2 = r2s.mean(0)
dat_adj = adjs.mean(0)
out.dat("adjr2", {"k": dat_k, "r2": dat_r2, "adj": dat_adj})
out.val("r2_0", dat_r2[0], 3)
out.val("adj_0", dat_adj[0], 3)
out.val("r2_20", dat_r2[20], 3)
out.val("adj_20", dat_adj[20], 3)
# OA: R^2 = 0.80 with n=50, p=4
out.val("oa_adj", 1 - (1 - 0.80) * (50 - 1) / (50 - 4 - 1), 4)

# ---------------------------------------------------------------- multicollinearity / VIF
N3 = 300
a = rng.normal(size=N3)
b = 0.95 * a + np.sqrt(1 - 0.95 ** 2) * rng.normal(size=N3)
c = rng.normal(size=N3)
F = np.column_stack([a, b, c])
vif_reg = []
for j in range(3):
    others = np.delete(F, j, axis=1)
    r2j = LinearRegression().fit(others, F[:, j]).score(others, F[:, j])
    vif_reg.append(1 / (1 - r2j))
Rinv = np.linalg.inv(np.corrcoef(F, rowvar=False))
out.check("VIF by auxiliary regressions vs diag(inv corr)", vif_reg, np.diag(Rinv))
out.tex("vif", ", ".join(f"{v:.2f}" for v in vif_reg))
out.val("vif_a", vif_reg[0], 1)
out.val("vif_c", vif_reg[2], 2)
out.val("corr_ab", np.corrcoef(a, b)[0, 1], 3)
# coefficient instability over bootstrap resamples
yy = 1 + a + b + c + rng.normal(0, 1, N3)
coefs = []
for _ in range(500):
    idx = rng.integers(0, N3, N3)
    coefs.append(LinearRegression().fit(F[idx], yy[idx]).coef_)
coefs = np.array(coefs)
out.val("sd_a", coefs[:, 0].std(), 3)
out.val("sd_c", coefs[:, 2].std(), 3)
out.val("sd_sum", (coefs[:, 0] + coefs[:, 1]).std(), 3)
out.val("ratio_sd", coefs[:, 0].std() / coefs[:, 2].std(), 1)

# ---------------------------------------------------------------- polynomial feature counts
for p_, d_ in [(2, 2), (3, 2), (10, 2), (10, 3), (100, 2)]:
    cnt = PolynomialFeatures(d_).fit(np.zeros((1, p_))).n_output_features_
    out.check(f"#poly features p={p_}, d={d_} = C(p+d,d)", cnt, comb(p_ + d_, d_))
    out.val(f"poly_{p_}_{d_}", cnt)

# ---------------------------------------------------------------- outlier: OLS vs Huber
xo = np.arange(10, dtype=float)
yo = 2 * xo + 1 + rng.normal(0, 0.5, 10)
yo[9] = 60.0
ols = LinearRegression().fit(xo[:, None], yo)
hub = HuberRegressor().fit(xo[:, None], yo)
out.val("ols_slope", ols.coef_[0], 2)
out.val("hub_slope", hub.coef_[0], 2)
grid = np.linspace(0, 9, 30)
out.dat("outlier_line", {"x": grid, "ols": ols.intercept_ + ols.coef_[0] * grid,
                         "hub": hub.intercept_ + hub.coef_[0] * grid})
out.dat("outlier_pts", {"x": xo, "y": yo})
