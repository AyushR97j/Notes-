def kmeans(X, C, iters=100):
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        Cn = np.array([X[lab == k].mean(0) for k in range(len(C))])
        if np.allclose(Cn, C):
            break
        C = Cn
    return C, lab
