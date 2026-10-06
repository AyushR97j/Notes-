def fit_l2(Phi_x, y, lam):
    """Minimise sum (y - b - Phi w)^2 + lam * ||w||^2 (bias not penalised)."""
    mu_x, mu_y = Phi_x.mean(0), y.mean()
    Xc = Phi_x - mu_x
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ (y - mu_y))
    return w, mu_y - mu_x @ w
