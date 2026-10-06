"""Numbers used in chapter 6's problems."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.naive_bayes import CategoricalNB
from common import sci, setup

out = setup(__file__)
s, h = 0.3 * 0.05 * 0.002, 0.7 * 0.005 * 0.01
out.val("p6a_s", sci(s)); out.val("p6a_h", sci(h)); out.val("p6a_post", s / (s + h), 4)
out.val("p6b_pos", 1 / 1500, 5); out.val("p6b_neg", 13 / 1800, 5); out.val("p6b_ratio", (13 / 1800) / (1 / 1500), 2)
x = brentq(lambda t: norm.logpdf(t, 0, 1) - norm.logpdf(t, 0, 2), 0.1, 5)
out.check("6D boundary sqrt(8 ln2 / 3)", x, np.sqrt(8 * np.log(2) / 3))
out.val("p6d_x", x, 3)
# 6C(a): naive Bayes on XOR is at chance
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25); y = np.array([0, 1, 1, 0] * 25)
out.check("6C NB on XOR predicts 0.5 for every input", CategoricalNB().fit(X, y).predict_proba(X)[:4, 1], [0.5] * 4)
