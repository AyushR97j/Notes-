class Embedding:
    def __init__(self, E):
        self.params = {"E": E.copy()}
    def forward(self, idx):                         # idx: (B, ctx) -> (B, ctx*dim)
        self.idx = idx
        return self.params["E"][idx].reshape(len(idx), -1)
    def backward(self, g):
        gE = np.zeros_like(self.params["E"])
        np.add.at(gE, self.idx, g.reshape(*self.idx.shape, -1))   # rows used twice get both gradients
        self.grads = {"E": gE}

class Tanh:
    params = {}
    def forward(self, x):
        self.y = np.tanh(x); return self.y
    def backward(self, g):
        return g * (1 - self.y ** 2)
