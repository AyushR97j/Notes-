def layer_norm(x, gamma, beta, eps=1e-5):         # normalise over the last axis
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def rms_norm(x, gamma, eps=1e-6):                 # no centring, no beta
    return gamma * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)

def group_norm(x, G, gamma, beta, eps=1e-5):      # x: (B, C, H, W)
    B, C, H, W = x.shape
    xg = x.reshape(B, G, C // G, H, W)
    mu = xg.mean((2, 3, 4), keepdims=True); var = xg.var((2, 3, 4), keepdims=True)
    xg = (xg - mu) / np.sqrt(var + eps)
    return xg.reshape(B, C, H, W) * gamma[None, :, None, None] + beta[None, :, None, None]
