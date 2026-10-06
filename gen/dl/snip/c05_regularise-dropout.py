def dropout(h, p, train, rng):
    if not train or p == 0:
        return h                                    # inference: identity
    mask = (rng.random(h.shape) >= p)               # keep with probability 1-p
    return h * mask / (1 - p)                       # inverted scaling
