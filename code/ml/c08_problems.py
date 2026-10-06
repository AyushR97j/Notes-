"""Numbers used in chapter 8's problems."""
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from common import setup

out = setup(__file__)
w, b, x = np.array([3.0, 4]), -2.0, np.array([1.0, 1])
out.val("p8a_w", 2 / np.linalg.norm(w), 2); out.val("p8a_f", w @ x + b, 0)
out.val("p8a_d", (w @ x + b) / np.linalg.norm(w), 2)
x, z = np.array([1.0, 0, 2]), np.array([2.0, 1, 1])
out.val("p8b_lin", x @ z, 0)
out.check("8B polynomial kernel vs sklearn", (x @ z + 1) ** 3, polynomial_kernel([x], [z], degree=3, gamma=1, coef0=1)[0, 0])
out.val("p8b_poly", (x @ z + 1) ** 3, 0)
out.check("8B RBF vs sklearn", np.exp(-0.1 * 3), rbf_kernel([x], [z], gamma=0.1)[0, 0])
out.val("p8b_rbf", np.exp(-0.3), 4)
lam, w, x, y = 0.1, np.array([0.5, -0.5]), np.array([1.0, 2]), 1
g = lam * w - (y * x if y * (w @ x) < 1 else 0)
w1 = w - 0.1 * g
out.val("p8e_g0", g[0], 2); out.val("p8e_g1", g[1], 2); out.val("p8e_w0", w1[0], 3); out.val("p8e_w1", w1[1], 3)
