class Tensor:
    def __init__(self, data, parents=(), backward=lambda g: ()):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.parents, self._backward = parents, backward

    @staticmethod
    def _unbroadcast(g, shape):             # sum gradient over broadcast dims
        while g.ndim > len(shape):
            g = g.sum(0)
        for i, s in enumerate(shape):
            if s == 1:
                g = g.sum(i, keepdims=True)
        return g

    def __add__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        return Tensor(self.data + o.data, (self, o),
                      lambda g: (self._unbroadcast(g, self.data.shape),
                                 self._unbroadcast(g, o.data.shape)))

    def __mul__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        return Tensor(self.data * o.data, (self, o),
                      lambda g: (self._unbroadcast(g * o.data, self.data.shape),
                                 self._unbroadcast(g * self.data, o.data.shape)))

    def __matmul__(self, o):
        return Tensor(self.data @ o.data, (self, o),
                      lambda g: (g @ o.data.T, self.data.T @ g))

    def relu(self):
        return Tensor(np.maximum(self.data, 0), (self,), lambda g: (g * (self.data > 0),))

    def tanh(self):
        t = np.tanh(self.data)
        return Tensor(t, (self,), lambda g: (g * (1 - t ** 2),))

    def sum(self):
        return Tensor(self.data.sum(), (self,), lambda g: (g * np.ones_like(self.data),))

    def cross_entropy(self, y):             # logits (B, K), integer labels (B,)
        Z = self.data - self.data.max(1, keepdims=True)
        P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        B = len(y)
        loss = -np.log(P[np.arange(B), y]).mean()
        def back(g):
            d = P.copy(); d[np.arange(B), y] -= 1
            return (g * d / B,)
        return Tensor(loss, (self,), back)

    def backward(self):
        order, seen = [], set()
        def visit(t):                       # topological order by DFS
            if id(t) not in seen:
                seen.add(id(t))
                for p in t.parents:
                    visit(p)
                order.append(t)
        visit(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):           # outputs before inputs
            for p, g in zip(t.parents, t._backward(t.grad)):
                p.grad = p.grad + g
