def perceptron(X, y, epochs=20, lr=1.0):
    w, b = np.zeros(X.shape[1]), 0.0
    log = []
    for ep in range(epochs):
        mistakes = 0
        for i in range(len(X)):
            if y[i] * (w @ X[i] + b) <= 0:      # mistake (or on the boundary)
                w, b = w + lr * y[i] * X[i], b + lr * y[i]
                mistakes += 1
                log.append((ep + 1, i + 1, w.copy(), b))
        if mistakes == 0:
            return w, b, log, ep + 1
    return w, b, log, epochs
