"""Numbers used in chapter 11's problems."""
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from common import setup

out = setup(__file__)
x = np.array([1.0, 2, 3, 8, 9, 10, 25])
km = KMeans(2, init=np.array([[2.0], [9.0]]), n_init=1, max_iter=1).fit(x[:, None])
c = np.sort(km.cluster_centers_.ravel())
out.check("11A one step vs sklearn", c, [2, 13])
out.val("p11a_c1", c[0], 0); out.val("p11a_c2", c[1], 0)
ev = np.array([4.0, 2.5, 1.0, 0.3, 0.2])
out.val("p11b_pc1", ev[0] / ev.sum(), 2)
w, V = np.linalg.eigh(np.array([[3.0, 1], [1, 3]]))
z = np.array([2.0, 0]) @ V[:, -1]
out.check("11C projection |z| = sqrt(2)", abs(z), np.sqrt(2)); out.val("p11c_z", abs(z), 4)
out.val("p11d_lift", (120 / 200) / (300 / 1000), 1)
d = np.array([0.3, 0.7]) * norm.pdf(1.5, [0, 4], 1)
out.val("p11e_r", d[0] / d.sum(), 4)
