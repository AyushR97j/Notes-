def gb_classify(X, y, M=100, nu=0.1, depth=3, seed=0, min_leaf=20):
    """Binary log-loss boosting; each leaf takes one Newton step: sum(y - p) / sum(p (1 - p))."""
    rs = np.random.RandomState(seed)
    X = X.astype(np.float32)
    F0 = np.log(y.mean() / (1 - y.mean()))
    F = np.full(len(y), F0)
    stages = []
    for _ in range(M):
        p = expit(F)
        r = y - p                                  # negative gradient of the log-loss
        t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=rs).fit(X, r)
        leaf = t.apply(X)
        vals = {}
        for l in np.unique(leaf):
            m = leaf == l
            vals[l] = r[m].sum() / (p[m] * (1 - p[m])).sum()
        F += nu * np.array([vals[l] for l in leaf])
        stages.append((t, vals))
    def decision(Z):
        out_ = np.full(len(Z), F0)
        for t, vals in stages:
            out_ += nu * np.array([vals[l] for l in t.apply(Z.astype(np.float32))])
        return out_
    return decision
