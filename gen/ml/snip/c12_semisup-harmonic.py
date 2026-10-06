def harmonic(X, y_semi, gamma):
    """Label propagation fixed point: f_u = (D_uu - W_uu)^{-1} W_ul Y_l (labels clamped)."""
    W = np.exp(-gamma * ((X[:, None] - X[None]) ** 2).sum(-1))
    L = y_semi >= 0
    Yl = np.eye(2)[y_semi[L]]
    D = np.diag(W.sum(1))
    Fu = np.linalg.solve((D - W)[np.ix_(~L, ~L)], W[np.ix_(~L, L)] @ Yl)
    F = np.zeros((len(X), 2)); F[L] = Yl; F[~L] = Fu
    return F
