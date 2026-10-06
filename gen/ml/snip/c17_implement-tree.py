def gini(counts):
    p = counts / counts.sum()
    return 1 - (p ** 2).sum()

def build_tree(X, y, w, depth, max_depth, n_classes):
    """w = sample weights (bootstrap counts for forests). Leaves store class weights."""
    counts = np.bincount(y, weights=w, minlength=n_classes)
    if depth == max_depth or gini(counts) == 0:
        return {"leaf": counts / counts.sum()}
    best = None
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        xs, ys, ws = X[order, j], y[order], w[order]
        left = np.cumsum(np.eye(n_classes)[ys] * ws[:, None], axis=0)   # class weights left of each cut
        total = left[-1]
        for i in np.where(xs[1:] > xs[:-1])[0]:                         # cut between distinct values only
            L, R = left[i], total - left[i]
            score = (L.sum() * gini(L) + R.sum() * gini(R)) / total.sum()
            if best is None or score < best[0] - 1e-12:
                best = (score, j, (xs[i] + xs[i + 1]) / 2)
    if best is None or best[0] >= gini(counts) - 1e-12:
        return {"leaf": counts / counts.sum()}
    _, j, t = best
    m = X[:, j] <= t
    return {"j": j, "t": t,
            "l": build_tree(X[m], y[m], w[m], depth + 1, max_depth, n_classes),
            "r": build_tree(X[~m], y[~m], w[~m], depth + 1, max_depth, n_classes)}

def tree_proba(node, x):
    while "leaf" not in node:
        node = node["l"] if x[node["j"]] <= node["t"] else node["r"]
    return node["leaf"]
