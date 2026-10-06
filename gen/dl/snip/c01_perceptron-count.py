def mlp_params(sizes):
    return sum(m * n + n for m, n in zip(sizes[:-1], sizes[1:]))
sizes = [784, 256, 128, 10]
n_ours = mlp_params(sizes)
