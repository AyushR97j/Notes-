def conv2d_single(X, K, stride=1, pad=0):
    """What deep-learning 'convolution' computes: cross-correlation (no flip)."""
    X = np.pad(X, pad)
    kh, kw = K.shape
    H = (X.shape[0] - kh) // stride + 1
    W = (X.shape[1] - kw) // stride + 1
    Y = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            patch = X[i * stride:i * stride + kh, j * stride:j * stride + kw]
            Y[i, j] = np.sum(patch * K)                 # elementwise product, then sum
    return Y
