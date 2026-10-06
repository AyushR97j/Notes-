"""Chapter 1: empirical risk, Bayes error, over/underfitting, selection optimism."""
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. empirical risk
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([1.2, 2.9, 5.1, 7.2, 8.7])
f = lambda t: 2 * t + 1
r = y - f(x)
out.val("er_sq", np.mean(r ** 2), 3)
out.val("er_abs", np.mean(np.abs(r)), 3)
out.tex("er_resid", " & ".join(f"{v:+.1f}" for v in r))
out.tex("er_pred", " & ".join(f"{v:.0f}" for v in f(x)))

# 0-1 risk of a threshold classifier on 6 labelled points
xc = np.array([0.5, 1.5, 2.2, 2.8, 3.5, 4.1])
yc = np.array([0, 0, 1, 0, 1, 1])
pred = (xc > 2.0).astype(int)
out.val("er01", np.mean(pred != yc), 4)
out.val("er01_wrong", int(np.sum(pred != yc)))

# ---------------------------------------------------------------- 2. Bayes error
# Class 0 ~ N(-1,1), class 1 ~ N(+1,1), equal priors -> threshold 0, error Phi(-1)
bayes = stats.norm.cdf(-1.0)
n = 200_000
lab = rng.integers(0, 2, n)
xs = rng.normal(2 * lab - 1, 1.0)
mc = np.mean((xs > 0).astype(int) != lab)
out.val("bayes_exact", bayes, 4)
out.val("bayes_mc", mc, 4)
out.check("Bayes error MC vs Phi(-1)", mc, bayes, atol=5e-3)
# unequal priors 0.8 / 0.2: threshold moves to ln(0.8/0.2)/2
p0, p1 = 0.8, 0.2
t = np.log(p0 / p1) / 2
err = p0 * (1 - stats.norm.cdf(t, -1, 1)) + p1 * stats.norm.cdf(t, 1, 1)
out.val("bayes_prior_t", t, 4)
out.val("bayes_prior_err", err, 4)
out.val("bayes_prior_naive", p0 * (1 - stats.norm.cdf(0, -1, 1)) + p1 * stats.norm.cdf(0, 1, 1), 4)

# ---------------------------------------------------------------- 3. polynomial degree sweep
def truth(t):
    return np.sin(2 * np.pi * t)

sigma = 0.3
ntr, nte, reps = 15, 2000, 200
degs = np.arange(0, 13)
tr_all = np.zeros((reps, len(degs)))
te_all = np.zeros((reps, len(degs)))
for k in range(reps):
    xtr = rng.uniform(0, 1, ntr)
    ytr = truth(xtr) + rng.normal(0, sigma, ntr)
    xte = rng.uniform(0, 1, nte)
    yte = truth(xte) + rng.normal(0, sigma, nte)
    for i, d in enumerate(degs):
        P = PolynomialFeatures(d)
        A = P.fit_transform(xtr[:, None])
        m = LinearRegression(fit_intercept=False).fit(A, ytr)
        tr_all[k, i] = np.mean((m.predict(A) - ytr) ** 2)
        te_all[k, i] = np.mean((m.predict(P.transform(xte[:, None])) - yte) ** 2)
# medians over the 200 repetitions: a few unlucky high-degree fits extrapolate
# wildly and would dominate a mean
tr_err = np.median(tr_all, axis=0)
te_err = np.median(te_all, axis=0)
out.val("te9_mean", te_all[:, 9].mean(), 0)
out.dat("degree", {"deg": degs, "train": tr_err, "test": np.minimum(te_err, 3.0)})
best = int(degs[np.argmin(te_err)])
out.val("deg_best", best)
out.val("noise_var", sigma ** 2, 2)
for d in (0, 1, 3, 9):
    out.val(f"tr{d}", tr_err[d], 3)
    out.val(f"te{d}", te_err[d], 3)
    out.val(f"gap{d}", te_err[d] - tr_err[d], 3)
out.val("te12", te_err[12], 2)

# one concrete fit for the figure (degree 1, 3, 9 on the same 15 points)
xtr = np.sort(rng.uniform(0, 1, ntr))
ytr = truth(xtr) + rng.normal(0, sigma, ntr)
grid = np.linspace(0, 1, 120)
cols = {"x": grid, "truth": truth(grid)}
for d in (1, 3, 9):
    P = PolynomialFeatures(d)
    m = LinearRegression(fit_intercept=False).fit(P.fit_transform(xtr[:, None]), ytr)
    cols[f"d{d}"] = np.clip(m.predict(P.transform(grid[:, None])), -2.5, 2.5)
out.dat("fits", cols)
out.dat("fitpts", {"x": xtr, "y": ytr})

# ---------------------------------------------------------------- 4. selection optimism
# 50 coin-flip "models" scored on a validation set of 100; keep the best
nval, nmodels, reps = 100, 50, 2000
best_val = np.empty(reps)
best_test = np.empty(reps)
for k in range(reps):
    yv = rng.integers(0, 2, nval)
    preds = rng.integers(0, 2, (nmodels, nval))
    accs = (preds == yv).mean(axis=1)
    best_val[k] = accs.max()
    best_test[k] = rng.binomial(10_000, 0.5) / 10_000  # fresh data: a coin is a coin
out.val("opt_val", best_val.mean(), 3)
out.val("opt_test", best_test.mean(), 3)
out.val("opt_nmodels", nmodels)
out.val("opt_nval", nval)
# analytic: expected max of 50 Binomial(100, .5)/100
ks = np.arange(0, 101)
cdf = stats.binom.cdf(ks, 100, 0.5)
pmf_max = cdf ** 50 - np.concatenate([[0], cdf[:-1]]) ** 50
emax = np.sum(ks / 100 * pmf_max)
out.val("opt_exact", emax, 3)
out.check("E[max of 50 val accuracies] simulation vs exact", best_val.mean(), emax, atol=3e-3)

# ---------------------------------------------------------------- 5. generalisation-gap OA numerics
out.val("oa_train", 0.98, 2)
out.val("oa_test", 0.81, 2)
out.val("oa_gap", 0.98 - 0.81, 2)
