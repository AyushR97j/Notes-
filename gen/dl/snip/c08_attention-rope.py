def rope(x, pos, base=10000.0):
    """Rotate consecutive pairs (x0,x1), (x2,x3), ... by angle pos * theta_j."""
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(pos * theta), np.sin(pos * theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = x1 * c - x2 * s, x1 * s + x2 * c
    return out
