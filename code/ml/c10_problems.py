"""Numbers used in chapter 10's problems."""
from itertools import product
from math import factorial

import numpy as np
from common import setup

out = setup(__file__)
out.val("p10a_a", (3 / 4) ** 4, 4); out.val("p10a_b", factorial(4) / 4 ** 4, 4)
out.val("p10a_c", 4 * (1 - (3 / 4) ** 4), 4)
samples = list(product(range(4), repeat=4))
out.check("10A(b) by enumeration of all 256 samples", np.mean([len(set(s)) == 4 for s in samples]), 24 / 256)
out.check("10A(c) by enumeration", np.mean([len(set(s)) for s in samples]), 4 * (1 - (3 / 4) ** 4))
out.val("p10b_alpha", 0.5 * np.log(4), 4)
w = np.full(5, 0.2); pred_ok = np.array([0, 1, 1, 1, 1], bool); a = 0.5 * np.log(4)
w = w * np.exp(np.where(pred_ok, -a, a)); w /= w.sum()
out.check("10B weights after the update", w, [0.5, 0.125, 0.125, 0.125, 0.125])
out.val("p10c_w", 4 / 6, 4)
out.val("p10d_gain", 0.5 * (1 / 2 + 1 / 1.5), 4)
out.val("p10f_10", 0.25 * 4 + 0.75 * 4 / 10, 2); out.val("p10f_inf", 0.25 * 4, 1); out.val("p10f_rho", 0.5 / 4, 3)
