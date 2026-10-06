def bce_with_logits(z, y):          # = -[y log s(z) + (1-y) log(1-s(z))], stable
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
