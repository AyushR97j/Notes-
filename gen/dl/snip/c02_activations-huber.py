def huber(r, delta=1.0):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))
