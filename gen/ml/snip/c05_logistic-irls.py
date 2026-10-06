def logistic_newton(X, y, iters=10):
    w = np.zeros(X.shape[1])
    hist = []
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y)                       # gradient of the negative log-likelihood
        H = X.T @ (X * (p * (1 - p))[:, None])     # Hessian  X^T S X,  S = diag(p(1-p))
        w = w - np.linalg.solve(H, grad)
        p = 1 / (1 + np.exp(-X @ w))
        hist.append(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return w, hist
