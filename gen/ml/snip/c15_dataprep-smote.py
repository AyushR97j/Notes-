def smote(X, y, k=5, ratio=1.0, seed=0):
    """Add synthetic minority points on segments between a minority point and one of its k minority neighbours."""
    r = np.random.default_rng(seed)
    Xm = X[y == 1]
    n_new = int(ratio * ((y == 0).sum() - len(Xm)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    i = r.integers(0, len(Xm), n_new)
    j = nn[i, r.integers(0, k, n_new)]
    lam = r.random((n_new, 1))
    Xs = Xm[i] + lam * (Xm[j] - Xm[i])
    return np.vstack([X, Xs]), np.r_[y, np.ones(n_new, int)]
