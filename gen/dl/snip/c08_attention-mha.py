def mha(X, Wq, Wk, Wv, Wo, bq, bk, bv, bo, h, mask=None):
    """X: (B, T, d); W*: (d, d) applied as X @ W.T + b (nn.Linear layout)."""
    B, T, d = X.shape
    dk = d // h
    split = lambda Y: Y.reshape(B, T, h, dk).transpose(0, 2, 1, 3)   # (B, h, T, dk)
    Qh, Kh, Vh = (split(X @ W.T + b) for W, b in ((Wq, bq), (Wk, bk), (Wv, bv)))
    Oh, A = attention(Qh, Kh, Vh, mask)                              # (B, h, T, dk)
    concat = Oh.transpose(0, 2, 1, 3).reshape(B, T, d)               # (B, T, d)
    return concat @ Wo.T + bo, A
