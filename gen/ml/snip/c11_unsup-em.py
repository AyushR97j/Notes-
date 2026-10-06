def em_step(x, pi, mu, var):
    # E-step: responsibilities r_ik = pi_k N(x_i|mu_k,var_k) / sum_j (...)
    dens = pi * norm.pdf(x[:, None], mu, np.sqrt(var))
    r = dens / dens.sum(1, keepdims=True)
    # M-step: weighted MLEs
    Nk = r.sum(0)
    pi = Nk / len(x)
    mu = (r * x[:, None]).sum(0) / Nk
    var = (r * (x[:, None] - mu) ** 2).sum(0) / Nk
    return pi, mu, var, np.log(dens.sum(1)).sum()
