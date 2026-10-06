class SGD:
    def __init__(self, lr, momentum=0.0, nesterov=False, weight_decay=0.0):
        self.lr, self.mu, self.nesterov, self.wd, self.buf = lr, momentum, nesterov, weight_decay, None
    def step(self, p, g):
        g = g + self.wd * p                          # L2 penalty folded into g
        if self.mu:
            self.buf = g if self.buf is None else self.mu * self.buf + g
            g = g + self.mu * self.buf if self.nesterov else self.buf
        return p - self.lr * g

class AdaGrad:
    def __init__(self, lr, eps=1e-10):
        self.lr, self.eps, self.G = lr, eps, 0.0
    def step(self, p, g):
        self.G = self.G + g * g                      # sum of all squared gradients
        return p - self.lr * g / (np.sqrt(self.G) + self.eps)

class RMSProp:
    def __init__(self, lr, alpha=0.99, eps=1e-8):
        self.lr, self.a, self.eps, self.v = lr, alpha, eps, 0.0
    def step(self, p, g):
        self.v = self.a * self.v + (1 - self.a) * g * g   # EMA of squared gradients
        return p - self.lr * g / (np.sqrt(self.v) + self.eps)

class Adam:
    def __init__(self, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.wd, self.decoupled, self.m, self.v, self.t = weight_decay, decoupled, 0.0, 0.0, 0
    def step(self, p, g):
        self.t += 1
        if self.decoupled:                           # AdamW: decay the weights directly
            p = p * (1 - self.lr * self.wd)
        else:                                        # Adam + L2: decay enters the gradient
            g = g + self.wd * p
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)     # bias correction
        v_hat = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
