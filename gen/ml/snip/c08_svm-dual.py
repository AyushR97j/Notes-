def svm_dual(K, y, C=np.inf):
    """max sum(a) - 1/2 a^T (yy^T * K) a  s.t.  0 <= a_i <= C, sum a_i y_i = 0."""
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K
    obj = lambda a: 0.5 * a @ Q @ a - a.sum()
    jac = lambda a: Q @ a - 1
    bounds = [(0, None if np.isinf(C) else C)] * n
    res = minimize(obj, np.zeros(n), jac=jac, bounds=bounds, method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x
