def knn_predict(Xtr, ytr, Xte, k=5):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)      # (n_test, n_train) squared distances
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.array([np.bincount(ytr[r]).argmax() for r in idx])
