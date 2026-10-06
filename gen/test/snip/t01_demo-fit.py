x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)
xb, yb = x.mean(), y.mean()
slope = ((x - xb) * (y - yb)).sum() / ((x - xb) ** 2).sum()
intercept = yb - slope * xb
