def temperature(z, T):
    return softmax(z / T)

def top_k(z, k):
    keep = np.argsort(-z)[:k]
    zz = np.full_like(z, -np.inf); zz[keep] = z[keep]
    return softmax(zz)

def top_p(z, p):                                   # nucleus sampling
    probs = softmax(z)
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    n = int(np.searchsorted(cum, p) + 1)           # smallest prefix with mass >= p
    zz = np.full_like(z, -np.inf); zz[order[:n]] = z[order[:n]]
    return softmax(zz)
