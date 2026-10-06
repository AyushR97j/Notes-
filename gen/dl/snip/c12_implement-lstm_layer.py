def lstm_forward(xs, Wih, Whh, bih, bhh):
    """xs: (T, B, D); weights in nn.LSTM layout (gates i, f, g, o). Returns all h_t."""
    T, B, _ = xs.shape
    H = Whh.shape[1]
    h, c = np.zeros((B, H)), np.zeros((B, H))
    hs = []
    for t in range(T):
        z = xs[t] @ Wih.T + h @ Whh.T + bih + bhh
        i, f, g, o = sig(z[:, :H]), sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sig(z[:, 3 * H:])
        c = f * c + i * g
        h = o * np.tanh(c)
        hs.append(h)
    return np.stack(hs), (h, c)
