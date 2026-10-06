def attention(Q, K, V, mask=None):
    """Q: (..., Tq, dk), K: (..., Tk, dk), V: (..., Tk, dv); mask True = blocked."""
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])     # (..., Tq, Tk) scores
    if mask is not None:
        S = np.where(mask, -np.inf, S)                         # block before the softmax
    A = softmax(S, axis=-1)                                    # each row sums to 1
    return A @ V, A
