def lasso_cd(X, y, lam, iters=500):
    """min 1/(2n)||y - Xw||^2 + lam ||w||_1 by cyclic coordinate descent."""
    n, d = X.shape
    w = np.zeros(d)
    col_sq = (X ** 2).sum(0) / n
    r = y - X @ w
    for _ in range(iters):
        for j in range(d):
            r += X[:, j] * w[j]                      # remove j's contribution
            rho = X[:, j] @ r / n
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / col_sq[j]
            r -= X[:, j] * w[j]
    return w
