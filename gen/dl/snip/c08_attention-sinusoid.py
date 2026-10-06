def sinusoidal(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / 10000 ** (2 * i / d)
    PE = np.zeros((T, d))
    PE[:, 0::2], PE[:, 1::2] = np.sin(angle), np.cos(angle)
    return PE
