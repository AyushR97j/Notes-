def im2col(x, k, s=1, p=0):
    """x: (B, C, H, W) -> columns (B, C*k*k, Ho*Wo), one column per output position."""
    B, C, H, W = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    cols = np.empty((B, C, k, k, Ho, Wo))
    for u in range(k):
        for v in range(k):
            cols[:, :, u, v] = x[:, :, u:u + s * Ho:s, v:v + s * Wo:s]
    return cols.reshape(B, C * k * k, Ho * Wo), (Ho, Wo)

def col2im(cols, shape, k, s=1, p=0):
    """Adjoint of im2col: scatter-add columns back to an image (used in backward)."""
    B, C, H, W = shape
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    cols = cols.reshape(B, C, k, k, Ho, Wo)
    x = np.zeros((B, C, H + 2 * p, W + 2 * p))
    for u in range(k):
        for v in range(k):
            x[:, :, u:u + s * Ho:s, v:v + s * Wo:s] += cols[:, :, u, v]
    return x[:, :, p:p + H, p:p + W]

class Conv2d:
    def __init__(self, W, b, s=1, p=0):            # W: (Cout, Cin, k, k)
        self.params, self.s, self.p = {"W": W.copy(), "b": b.copy()}, s, p
    def forward(self, x):
        Cout, Cin, k, _ = self.params["W"].shape
        self.shape, self.k = x.shape, k
        self.cols, (Ho, Wo) = im2col(x, k, self.s, self.p)
        y = self.params["W"].reshape(Cout, -1) @ self.cols + self.params["b"][:, None]   # (B, Cout, Ho*Wo)
        return y.reshape(x.shape[0], Cout, Ho, Wo)
    def backward(self, g):                          # g: (B, Cout, Ho, Wo)
        B, Cout = g.shape[:2]
        g2 = g.reshape(B, Cout, -1)
        self.grads = {"W": np.einsum("bop,bcp->oc", g2, self.cols).reshape(self.params["W"].shape),
                      "b": g2.sum((0, 2))}
        dcols = np.einsum("oc,bop->bcp", self.params["W"].reshape(Cout, -1), g2)
        return col2im(dcols, self.shape, self.k, self.s, self.p)
