"""Numbers used in chapter 14's problems."""
import numpy as np
from scipy import stats
from common import setup

out = setup(__file__)
out.val("p14a_0", stats.poisson.pmf(0, 3), 4); out.val("p14a_2", 1 - stats.poisson.cdf(1, 3), 4)
t = (210 - 200) / (40 / 4); tc = stats.t.ppf(0.975, 15)
out.val("p14c_t", t, 2); out.val("p14c_p", 2 * stats.t.sf(t, 15), 4)
out.val("p14c_lo", 210 - tc * 10, 1); out.val("p14c_hi", 210 + tc * 10, 1)
def n_ab(p1, p2, a=0.05, pw=0.8):
    za, zb = stats.norm.ppf(1 - a / 2), stats.norm.ppf(pw); pb = (p1 + p2) / 2
    return int(np.ceil(((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1)) ** 2))
out.val("p14d_n", n_ab(0.10, 0.11))
out.check("14D ratio of sample sizes is about 4", n_ab(0.10, 0.11) / n_ab(0.10, 0.12), 4.0, atol=0.4)
