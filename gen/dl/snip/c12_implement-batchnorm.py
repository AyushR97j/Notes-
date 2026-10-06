class BatchNorm1d:
    def __init__(self, C, momentum=0.1, eps=1e-5):
        self.params = {"gamma": np.ones(C), "beta": np.zeros(C)}
        self.rm, self.rv, self.mom, self.eps, self.training = np.zeros(C), np.ones(C), momentum, eps, True
    def forward(self, x):
        if self.training:
            mu, var = x.mean(0), x.var(0)
            B = len(x)
            self.rm = (1 - self.mom) * self.rm + self.mom * mu
            self.rv = (1 - self.mom) * self.rv + self.mom * var * B / (B - 1)   # unbiased
        else:
            mu, var = self.rm, self.rv
        self.std = np.sqrt(var + self.eps)
        self.xhat = (x - mu) / self.std
        return self.params["gamma"] * self.xhat + self.params["beta"]
    def backward(self, g):
        B = len(g)
        self.grads = {"gamma": (g * self.xhat).sum(0), "beta": g.sum(0)}
        dxh = g * self.params["gamma"]
        return (B * dxh - dxh.sum(0) - self.xhat * (dxh * self.xhat).sum(0)) / (B * self.std)
