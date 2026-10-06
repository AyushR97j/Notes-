"""Numbers used in chapter 7's problems."""
import numpy as np
from scipy.spatial import distance as D
from common import setup

out = setup(__file__)
out.val("p7b_1", 0.01 ** (1 / 20), 3); out.val("p7b_50", 0.5 ** (1 / 20), 3)
A, B, C = np.array([2.0, 0, 1]), np.array([20.0, 0, 10]), np.array([1.0, 1, 1])
out.val("p7d_ab", D.euclidean(A, B), 3); out.val("p7d_ac", D.euclidean(A, C), 3)
out.val("p7d_cos", D.cosine(A, C), 3)
out.check("7D cosine(A,B) = 0", D.cosine(A, B), 0.0, atol=1e-12)
