x = np.array([0.5, -1.0, 2.0])
w = np.array([0.4, 0.3, -0.2])
b = 0.1
z = w @ x + b                                   # weighted sum (pre-activation)
acts = {
    "step":    float(z >= 0),
    "sigmoid": 1 / (1 + np.exp(-z)),
    "tanh":    np.tanh(z),
    "relu":    max(0.0, z),
}
