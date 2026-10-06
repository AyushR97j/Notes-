def relu_net_interp(f, n):
    """One hidden ReLU layer with n units that interpolates f at n+1 knots."""
    knots = np.linspace(0, 1, n + 1)
    vals = f(knots)
    slopes = np.diff(vals) / np.diff(knots)
    a = np.r_[slopes[0], np.diff(slopes)]           # output weights
    def net(t):
        H = np.maximum(0.0, t[:, None] - knots[:-1][None, :])   # hidden layer
        return vals[0] + H @ a
    return net
