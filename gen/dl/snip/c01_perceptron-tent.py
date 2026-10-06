def tent_layer(t):            # 2 ReLU units: g(t) = 2 relu(t) - 4 relu(t - 1/2)
    return 2 * np.maximum(0, t) - 4 * np.maximum(0, t - 0.5)
def deep_tent(t, L):
    for _ in range(L):
        t = tent_layer(t)
    return t
