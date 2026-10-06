def lda_scores(X, mus, Sigma, priors):
    """Linear discriminant: delta_k(x) = x^T S^-1 mu_k - mu_k^T S^-1 mu_k / 2 + log pi_k."""
    Si = np.linalg.inv(Sigma)
    return np.column_stack([X @ Si @ m - 0.5 * m @ Si @ m + np.log(p) for m, p in zip(mus, priors)])
