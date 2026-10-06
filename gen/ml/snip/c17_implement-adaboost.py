def adaboost(X, y, M):
    """y in {-1, +1}. Stumps are our CART trees of depth 1 fitted with sample weights."""
    w = np.full(len(y), 1 / len(y)); stumps, alphas = [], []
    yy = (y > 0).astype(int)
    for _ in range(M):
        T = build_tree(X, yy, w, 0, 1, 2)
        pred = np.where([tree_proba(T, x)[1] > 0.5 for x in X], 1, -1)
        err = w[pred != y].sum() / w.sum()
        a = np.log((1 - err) / err)                      # SAMME weight (= 2 x the 1/2-log convention)
        w = w * np.exp(a * (pred != y)); w /= w.sum()
        stumps.append(T); alphas.append(a)
    return stumps, np.array(alphas)
