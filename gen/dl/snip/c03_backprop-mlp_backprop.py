def mlp_forward_backward(X, y, Ws, bs):
    """ReLU hidden layers, softmax + cross-entropy output. X: (B, n0), y: (B,) ints.
    Ws[l]: (n_{l+1}, n_l) as in nn.Linear.  Returns loss and gradients."""
    B = X.shape[0]
    hs, zs = [X], []
    for l, (W, b) in enumerate(zip(Ws, bs)):
        z = hs[-1] @ W.T + b                      # (B, n_{l+1})
        zs.append(z)
        hs.append(np.maximum(z, 0) if l < len(Ws) - 1 else z)
    Z = zs[-1] - zs[-1].max(1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(B), y]))
    delta = P.copy(); delta[np.arange(B), y] -= 1; delta /= B      # dL/dz_L
    gWs, gbs = [None] * len(Ws), [None] * len(Ws)
    for l in reversed(range(len(Ws))):
        gWs[l] = delta.T @ hs[l]                  # (n_{l+1}, n_l)
        gbs[l] = delta.sum(0)
        if l > 0:
            delta = (delta @ Ws[l]) * (zs[l - 1] > 0)  # back through W, then ReLU
    return loss, gWs, gbs
