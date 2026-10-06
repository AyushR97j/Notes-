"""Numbers used in chapter 4's problems."""
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from common import setup

out = setup(__file__)
soft = lambda z, t: np.sign(z) * max(abs(z) - t, 0)
for lam in (1, 3):
    out.val(f"p4b_lasso{lam}", soft(2.4, lam) + 0.0, 1)
    out.val(f"p4b_ridge{lam}", 2.4 / (1 + lam), 1)
# check the 1-D formulas with sklearn on a unit-norm feature (sum x^2 = 1)
rng = np.random.default_rng(0)
x = rng.normal(size=50); x /= np.linalg.norm(x)
y = 2.4 * x + 0.3 * rng.normal(size=50); y -= (x @ y - 2.4) * x  # force OLS coef = 2.4
out.check("OLS coefficient is 2.4", x @ y, 2.4)
out.check("ridge 1-D, lambda=3", Ridge(alpha=3, fit_intercept=False).fit(x[:, None], y).coef_[0], 0.6)
out.check("lasso 1-D, threshold 1", Lasso(alpha=1 / 50, fit_intercept=False, tol=1e-12).fit(x[:, None], y).coef_[0], 1.4)
# 4E: MSE of c * mean, simulation vs formula
mu, sig, n, c = 1.0, 2.0, 10, 0.7
means = rng.normal(mu, sig, size=(200000, n)).mean(1)
out.check("MSE(c xbar) simulation vs (1-c)^2 mu^2 + c^2 s^2/n", np.mean((c * means - mu) ** 2),
          (1 - c) ** 2 * mu ** 2 + c ** 2 * sig ** 2 / n, rtol=0.02)
