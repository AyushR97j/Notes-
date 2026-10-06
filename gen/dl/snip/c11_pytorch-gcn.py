def gcn_layer(A, H, W):
    A_hat = A + np.eye(len(A))                       # add self-loops
    d = A_hat.sum(1)
    A_norm = A_hat / np.sqrt(d[:, None] * d[None, :])  # D^-1/2 (A+I) D^-1/2
    return np.maximum(0, A_norm @ H @ W)             # aggregate neighbours, transform, ReLU
