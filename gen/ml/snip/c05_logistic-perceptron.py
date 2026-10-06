def perceptron(X, y, epochs=20, lr=1.0):
    """y in {-1,+1}; X includes a bias column. Returns weights and #mistakes per epoch."""
    w = np.zeros(X.shape[1])
    mistakes = []
    for _ in range(epochs):
        m = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi) <= 0:      # misclassified (or on the boundary)
                w += lr * yi * xi
                m += 1
        mistakes.append(m)
        if m == 0:
            break
    return w, mistakes
