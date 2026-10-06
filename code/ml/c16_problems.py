"""Numbers used in chapter 16's problems."""
import numpy as np
from scipy.optimize import minimize_scalar
from common import setup

out = setup(__file__)
x = 1.0; xs = []
for _ in range(2):
    x = x - (4 * x ** 3 - 3) / (12 * x ** 2); xs.append(x)
out.val("p16b_x1", xs[0], 4); out.val("p16b_x2", xs[1], 4)
star = minimize_scalar(lambda t: t ** 4 - 3 * t).x
out.check("16B minimiser (3/4)^(1/3) vs scipy", star, 0.75 ** (1 / 3), atol=1e-6)
out.val("p16b_star", 0.75 ** (1 / 3), 4)
