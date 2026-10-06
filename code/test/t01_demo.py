import numpy as np
from sklearn.linear_model import LinearRegression
from common import setup

out = setup(__file__)
# [[fit]]
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)
xb, yb = x.mean(), y.mean()
slope = ((x - xb) * (y - yb)).sum() / ((x - xb) ** 2).sum()
intercept = yb - slope * xb
# [[/fit]]
ref = LinearRegression().fit(x[:, None], y)
out.check("slope", slope, ref.coef_[0])
out.check("intercept", intercept, ref.intercept_)
out.val("slope", slope, 2)
out.val("intercept", intercept, 2)
xs = np.linspace(0, 6, 25)
out.dat("line", {"x": xs, "y": intercept + slope * xs})
out.dat("pts", {"x": x, "y": y})
out.text("print", f"slope={slope:.2f} intercept={intercept:.2f}")
