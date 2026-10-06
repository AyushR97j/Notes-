def forest_proba(Xtr, ytr, Xte, bootstrap_indices, max_depth):
    P = 0
    for idx in bootstrap_indices:
        w = np.bincount(idx, minlength=len(ytr)).astype(float)       # bootstrap = integer sample weights
        keep = w > 0
        T = build_tree(Xtr[keep], ytr[keep], w[keep], 0, max_depth, 2)
        P = P + np.array([tree_proba(T, x) for x in Xte])
    return P / len(bootstrap_indices)
