def step_decay(t, lr0, step=30, gamma=0.1):
    return lr0 * gamma ** (t // step)

def cosine(t, lr0, T, lr_min=0.0):
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * t / T))

def warmup_cosine(t, lr0, W, T):
    if t < W:
        return lr0 * (t + 1) / W                     # linear warm-up
    return cosine(t - W, lr0, T - W)

def one_cycle(t, lr_max, T, pct=0.3, div=25.0, final_div=1e4):
    up = pct * T - 1
    lo, end = lr_max / div, lr_max / div / final_div
    cos_anneal = lambda a, b, frac: b + (a - b) / 2 * (1 + math.cos(math.pi * frac))
    if t <= up:
        return cos_anneal(lo, lr_max, t / up)
    return cos_anneal(lr_max, end, (t - up) / (T - 1 - up))
