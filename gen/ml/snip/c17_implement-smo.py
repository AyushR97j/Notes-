def svm_smo(X, y, C=1.0, tol=1e-6, max_iter=100000):
    """Dual coordinate ascent on pairs (the working-set rule of LIBSVM's first version)."""
    n = len(y); K = X @ X.T; a = np.zeros(n); g = -np.ones(n)      # g = gradient of 1/2 a^T Q a - sum a
    Q = (y[:, None] * y[None, :]) * K
    for _ in range(max_iter):
        up = ((y > 0) & (a < C)) | ((y < 0) & (a > 0))            # can move y_i a_i up
        dn = ((y > 0) & (a > 0)) | ((y < 0) & (a < C))
        i = np.where(up)[0][np.argmax(-y[up] * g[up])]
        j = np.where(dn)[0][np.argmin(-y[dn] * g[dn])]
        if -y[i] * g[i] + y[j] * g[j] < tol:                      # KKT satisfied within tol
            break
        # move along y_i d_i = -y_j d_j that decreases the objective, then clip to the box
        quad = max(Q[i, i] + Q[j, j] - 2 * y[i] * y[j] * Q[i, j], 1e-12)
        step = (-y[i] * g[i] + y[j] * g[j]) / quad
        lo = max(-a[i] if y[i] > 0 else a[i] - C, a[j] - C if y[j] > 0 else -a[j])
        hi = min(C - a[i] if y[i] > 0 else a[i], a[j] if y[j] > 0 else C - a[j])
        step = min(max(step, lo), hi)
        di, dj = y[i] * step, -y[j] * step
        a[i] += di; a[j] += dj
        g += Q[:, i] * di + Q[:, j] * dj
    w = (a * y) @ X
    free = (a > 1e-8) & (a < C - 1e-8)
    b = np.mean(y[free] - X[free] @ w)
    return w, b, a
