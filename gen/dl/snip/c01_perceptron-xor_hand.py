step = lambda t: (t >= 0).astype(float)
def xor_net(X):
    h1 = step(X @ np.array([1, 1]) - 0.5)     # OR
    h2 = step(X @ np.array([1, 1]) - 1.5)     # AND
    return step(h1 - h2 - 0.5), h1, h2        # OR and not AND
