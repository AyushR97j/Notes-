def kl_gauss_std(mu, logvar):          # KL( N(mu, sigma^2) || N(0, 1) ), per dimension
    return 0.5 * (mu ** 2 + np.exp(logvar) - 1 - logvar)
