def lstm_cell(x, h, c, W, U, b):
    """W: (4H, D), U: (4H, H), b: (4H,), gates stacked in PyTorch order i, f, g, o."""
    z = W @ x + U @ h + b
    i, f, g, o = np.split(z, 4)
    i, f, o = sig(i), sig(f), sig(o)          # input, forget, output gates in (0, 1)
    g = np.tanh(g)                            # candidate cell update
    c = f * c + i * g                         # cell state: forget some, write some
    h = o * np.tanh(c)                        # hidden state: expose some of the cell
    return h, c
