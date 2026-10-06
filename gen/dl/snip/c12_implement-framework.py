class Linear:
    def __init__(self, W, b):                      # W: (out, in) as in nn.Linear
        self.params = {"W": W.copy(), "b": b.copy()}
    def forward(self, x):
        self.x = x
        return x @ self.params["W"].T + self.params["b"]
    def backward(self, g):                         # g = dL/d(output), shape (B, out)
        self.grads = {"W": g.T @ self.x, "b": g.sum(0)}
        return g @ self.params["W"]

class ReLU:
    params = {}
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, g):
        return g * self.mask

class Sequential:
    def __init__(self, *layers):
        self.layers = layers
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    def backward(self, g):
        for l in reversed(self.layers):
            g = l.backward(g)
        return g

def softmax_ce(logits, y):                          # mean cross-entropy and its gradient
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    B = len(y)
    loss = -np.log(p[np.arange(B), y]).mean()
    g = p.copy(); g[np.arange(B), y] -= 1
    return loss, g / B

class Adam:
    def __init__(self, layers, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.layers, self.lr, self.b1, self.b2, self.eps, self.t = layers, lr, b1, b2, eps, 0
        self.m = [{k: np.zeros_like(v) for k, v in l.params.items()} for l in layers]
        self.v = [{k: np.zeros_like(v) for k, v in l.params.items()} for l in layers]
    def step(self):
        self.t += 1
        for l, m, v in zip(self.layers, self.m, self.v):
            for k in l.params:
                g = l.grads[k]
                m[k] = self.b1 * m[k] + (1 - self.b1) * g
                v[k] = self.b2 * v[k] + (1 - self.b2) * g * g
                mh, vh = m[k] / (1 - self.b1 ** self.t), v[k] / (1 - self.b2 ** self.t)
                l.params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)
