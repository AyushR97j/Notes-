"""Numbers used in chapter 5's problems."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss
from common import setup

out = setup(__file__)
z = 0.5 * 2 - 1.0 * 1 + 0.25
out.val("p5a_z", z, 2); out.val("p5a_p", expit(z), 4); out.val("p5a_loss", -np.log(expit(z)), 4)
out.check("5A loss vs sklearn", -np.log(expit(z)), log_loss([1], [[1 - expit(z), expit(z)]], labels=[0, 1]))
odds = 0.25 * np.exp(1.2)
out.val("p5b_or", np.exp(1.2), 4); out.val("p5b_odds", odds, 4); out.val("p5b_p", odds / (1 + odds), 4)
X = np.array([[1, 1.0], [1, -1.0]]); y = np.array([0, 1]); w = np.array([0.0, 1.0])
p = expit(X @ w); e = p - y; g = X.T @ e / 2; w1 = w - 0.5 * g
out.val("p5c_p1", p[0], 4); out.val("p5c_p2", p[1], 4)
out.val("p5c_e1", e[0], 4); out.val("p5c_e2", e[1], 4)
out.val("p5c_g0", g[0], 4); out.val("p5c_g1", g[1], 4)
out.val("p5c_w0", w1[0], 4); out.val("p5c_w1", w1[1], 4)
out.val("p5f_t", 100 / 5100, 4)
