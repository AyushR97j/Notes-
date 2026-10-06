def pca(X, k):
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k] * np.sign(Vt[:k, [0]])            # fix the sign convention
    return Xc @ comps.T, comps, s[:k] ** 2 / (len(X) - 1)
