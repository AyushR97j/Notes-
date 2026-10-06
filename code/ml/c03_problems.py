"""Numbers used in the end-of-chapter problems of chapters 1-3."""
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from common import setup

out = setup(__file__)

# 1B: MSE / MAE
yh = np.array([3.0, 5, 2, 8]); y = np.array([2.0, 5, 4, 7])
out.val("p1b_mse", np.mean((yh - y) ** 2), 1)
out.val("p1b_mae", np.mean(np.abs(yh - y)), 0)
y2 = y.copy(); y2[-1] = 17
out.val("p1b_mse2", np.mean((yh - y2) ** 2), 1)
out.val("p1b_mae2", np.mean(np.abs(yh - y2)), 0)
# 1F: normal approximation to the best of 50 coin-flip models
emax = np.random.default_rng(0).normal(size=(200000, 50)).max(1).mean()
out.val("p1f_emax", emax, 2)
out.val("p1f_best", 0.5 + emax * 0.05, 3)

# 2A
out.val("p2a_1", 0.3 ** 2, 2); out.val("p2a_2", 0.6 ** 2, 2)
# 2D: two positives
odds = (0.95 / 0.10) ** 2 / 99
out.val("p2d_odds", odds, 4)
out.val("p2d_post", odds / (1 + odds), 3)
prior = 0.01
post1 = 0.95 * prior / (0.95 * prior + 0.1 * (1 - prior))
post2 = 0.95 * post1 / (0.95 * post1 + 0.1 * (1 - post1))
out.check("2D odds form vs sequential Bayes", odds / (1 + odds), post2)
# 2E: CTR
out.val("p2e_mle", 2 / 3, 3)
out.val("p2e_map", (2 + 2 - 1) / (3 + 2 + 20 - 2), 3)

# 3A: five points
x = np.array([1.0, 2, 3, 4, 5]); y = np.array([1.0, 3, 2, 5, 4])
m = LinearRegression().fit(x[:, None], y)
out.val("p3a_b1", m.coef_[0], 1); out.val("p3a_b0", m.intercept_, 1)
yhat = m.predict(x[:, None])
out.tex("p3a_yhat", ", ".join(f"{v:.1f}" for v in yhat))
out.tex("p3a_res", ", ".join(f"{v:.1f}" for v in y - yhat))
out.val("p3a_rss", ((y - yhat) ** 2).sum(), 1)
out.val("p3a_tss", ((y - y.mean()) ** 2).sum(), 0)
out.val("p3a_r2", m.score(x[:, None], y), 2)
out.val("p3a_pred7", m.predict([[7.0]])[0], 1)
# 3B summary statistics
out.val("p3b_b1", 0.6 * 10 / 2, 0); out.val("p3b_b0", 60 - 3 * 5, 0)
out.val("p3b_pred", 45 + 3 * 8, 0); out.val("p3b_rev", 0.6 * 2 / 10, 2)
# check via simulated data with exactly these moments
rng = np.random.default_rng(1)
z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=100)
z = (z - z.mean(0)) / z.std(0, ddof=1)
c = np.corrcoef(z.T)[0, 1]
hrs = 5 + 2 * z[:, 0]; sc = 60 + 10 * z[:, 1]
out.check("3B slope = r s_y/s_x on simulated data", LinearRegression().fit(hrs[:, None], sc).coef_[0],
          c * 10 / 2)
# 3D adjusted R^2
adj = lambda r2, n, p: 1 - (1 - r2) * (n - 1) / (n - p - 1)
out.val("p3d_a", adj(0.70, 60, 3), 4); out.val("p3d_b", adj(0.74, 60, 10), 4)
