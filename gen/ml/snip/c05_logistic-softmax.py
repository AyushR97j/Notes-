def softmax_regression(X, Y, lam, lr=0.5, iters=20000):
    """min  -1/n sum log p_{i,y_i} + lam/2 ||W[1:]||^2  by gradient descent."""
    n, d = X.shape
    W = np.zeros((d, Y.shape[1]))
    for _ in range(iters):
        Z = X @ W
        Z -= Z.max(axis=1, keepdims=True)           # stable softmax
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        G = X.T @ (P - Y) / n                       # the famous (P - Y)
        G[1:] += lam * W[1:]                        # do not penalise the intercept row
        W -= lr * G
    return W
