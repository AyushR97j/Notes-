def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)        # shift: same result, no overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def logsumexp(z):
    m = z.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True))).squeeze(-1)

def cross_entropy(z, y):                         # z: (B, K) logits, y: (B,) ints
    return np.mean(logsumexp(z) - z[np.arange(len(y)), y])
