def fit_linear(X, y, lam=0.0):
    """OLS (lam=0) or ridge; the intercept is not penalised (fit on centred data)."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w, ym - xm @ w
