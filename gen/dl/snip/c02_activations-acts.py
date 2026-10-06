def sigmoid(z):  return 1 / (1 + np.exp(-z))
def relu(z):     return np.maximum(0, z)
def leaky(z, a=0.01): return np.where(z > 0, z, a * z)
def elu(z, a=1.0):    return np.where(z > 0, z, a * (np.exp(z) - 1))
def gelu(z):     return z * norm.cdf(z)                       # exact: z * Phi(z)
def gelu_tanh(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))
def silu(z):     return z * sigmoid(z)                        # a.k.a. swish
def softplus(z): return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
# derivatives
def d_sigmoid(z): s = sigmoid(z); return s * (1 - s)
def d_tanh(z):    return 1 - np.tanh(z) ** 2
def d_relu(z):    return (z > 0).astype(float)
def d_gelu(z):    return norm.cdf(z) + z * norm.pdf(z)
def d_silu(z):    s = sigmoid(z); return s * (1 + z * (1 - s))
