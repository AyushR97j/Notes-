def numerical_grad(f, theta, h=1e-5):
    g = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta); e.flat[i] = h
        g.flat[i] = (f(theta + e) - f(theta - e)) / (2 * h)      # centred difference
    return g

def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b)))
