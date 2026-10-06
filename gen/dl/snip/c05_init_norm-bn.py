def bn_forward(x, gamma, beta, eps=1e-5):          # x: (B, C), training mode
    mu = x.mean(0)
    var = x.var(0)                                 # biased (divide by B)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, var, eps)

def bn_backward(dy, gamma, cache):
    xhat, var, eps = cache
    B = dy.shape[0]
    dgamma = (dy * xhat).sum(0)
    dbeta = dy.sum(0)
    dxhat = dy * gamma
    dx = (B * dxhat - dxhat.sum(0) - xhat * (dxhat * xhat).sum(0)) / (B * np.sqrt(var + eps))
    return dx, dgamma, dbeta
