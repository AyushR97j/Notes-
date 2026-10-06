def q_sample(x0, t, eps):              # closed form of t noising steps at once
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps
