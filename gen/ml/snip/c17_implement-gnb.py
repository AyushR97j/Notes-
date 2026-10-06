def gnb_fit_predict_proba(Xtr, ytr, Xte, eps=1e-9):
    classes = np.unique(ytr)
    eps = eps * Xtr.var(0).max()                                  # sklearn's var_smoothing convention
    logp = []
    for c in classes:
        Xk = Xtr[ytr == c]
        mu, var = Xk.mean(0), Xk.var(0) + eps
        ll = -0.5 * (np.log(2 * np.pi * var) + (Xte - mu) ** 2 / var).sum(1)
        logp.append(np.log(len(Xk) / len(Xtr)) + ll)
    L = np.array(logp).T
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)
