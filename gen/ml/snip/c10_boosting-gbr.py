def gb_regress(X, y, M=100, nu=0.1, depth=3, seed=0):
    rs = np.random.RandomState(seed)               # one RNG shared by all trees, as sklearn does
    X = X.astype(np.float32)                       # sklearn grows its trees on float32 inputs
    F0 = y.mean()
    F = np.full(len(y), F0)
    trees = []
    for _ in range(M):
        r = y - F                                  # negative gradient of 1/2 (y - F)^2
        t = DecisionTreeRegressor(max_depth=depth, random_state=rs).fit(X, r)
        F += nu * t.predict(X)
        trees.append(t)
    return lambda Z: F0 + nu * sum(t.predict(Z.astype(np.float32)) for t in trees)
