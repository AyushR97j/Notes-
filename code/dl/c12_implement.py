"""Chapter 12: implement it. A tiny NumPy framework (Linear, ReLU, BatchNorm1d,
Embedding, softmax cross-entropy, Adam) whose training trajectories are asserted
equal to PyTorch's; conv2d forward/backward via im2col; an LSTM layer over a
sequence; a character-level language model trained in NumPy and in PyTorch."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from common import setup

out = setup(__file__)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)


# ============================================================ the framework
# [[framework]]
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
# [[/framework]]

# [[batchnorm]]
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
# [[/batchnorm]]


# ------------------------- MLP with BatchNorm: NumPy vs PyTorch, 50 Adam steps
X, Y = make_classification(256, 10, n_informative=6, n_classes=3, random_state=0)
torch.manual_seed(0)
tnet = nn.Sequential(nn.Linear(10, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 3))
sd = {k: v.detach().numpy().copy() for k, v in tnet.state_dict().items()}
l1 = Linear(sd["0.weight"], sd["0.bias"]); bn = BatchNorm1d(32); l2 = Linear(sd["3.weight"], sd["3.bias"])
net = Sequential(l1, bn, ReLU(), l2)
opt = Adam([l1, bn, l2], lr=0.01)
topt = torch.optim.Adam(tnet.parameters(), lr=0.01)
Xt, Yt = torch.tensor(X), torch.tensor(Y)
g = np.random.default_rng(1)
ours_l, ref_l = [], []
for step in range(50):
    idx = g.choice(256, 32, replace=False)
    loss, gl = softmax_ce(net.forward(X[idx]), Y[idx]); net.backward(gl); opt.step()
    topt.zero_grad(); lt = F.cross_entropy(tnet(Xt[idx]), Yt[idx]); lt.backward(); topt.step()
    ours_l.append(loss); ref_l.append(lt.item())
out.check("NumPy MLP+BN+Adam: 50-step loss trajectory vs PyTorch", ours_l, ref_l, atol=1e-9)
out.check("NumPy vs PyTorch weights after 50 steps", l2.params["W"], tnet[3].weight.detach().numpy(), atol=1e-9)
out.check("BN running variance after 50 steps", bn.rv, tnet[1].running_var.numpy(), atol=1e-9)
bn.training = False; tnet.eval()
out.check("eval-mode outputs (running statistics)", net.forward(X[:20]), tnet(Xt[:20]).detach().numpy(), atol=1e-9)
out.val("mlp_loss0", ours_l[0], 4); out.val("mlp_loss49", ours_l[-1], 4)
out.val("mlp_acc", 100 * np.mean(net.forward(X).argmax(1) == Y), 1)
# softmax regression = one Linear layer, gradient-checked
W0 = rng.normal(size=(3, 10)) * 0.1; b0 = np.zeros(3)
lin = Linear(W0, b0)
loss, gl = softmax_ce(lin.forward(X), Y); lin.backward(gl)
num = np.zeros_like(W0); h = 1e-6
for i in range(3):
    for j in range(10):
        Wp = W0.copy(); Wp[i, j] += h; Wm = W0.copy(); Wm[i, j] -= h
        num[i, j] = (softmax_ce(X @ Wp.T, Y)[0] - softmax_ce(X @ Wm.T, Y)[0]) / (2 * h)
out.check("softmax regression gradient vs centred differences", lin.grads["W"], num, atol=1e-7)


# ===================================================================== conv
# [[im2col]]
def im2col(x, k, s=1, p=0):
    """x: (B, C, H, W) -> columns (B, C*k*k, Ho*Wo), one column per output position."""
    B, C, H, W = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    cols = np.empty((B, C, k, k, Ho, Wo))
    for u in range(k):
        for v in range(k):
            cols[:, :, u, v] = x[:, :, u:u + s * Ho:s, v:v + s * Wo:s]
    return cols.reshape(B, C * k * k, Ho * Wo), (Ho, Wo)

def col2im(cols, shape, k, s=1, p=0):
    """Adjoint of im2col: scatter-add columns back to an image (used in backward)."""
    B, C, H, W = shape
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    cols = cols.reshape(B, C, k, k, Ho, Wo)
    x = np.zeros((B, C, H + 2 * p, W + 2 * p))
    for u in range(k):
        for v in range(k):
            x[:, :, u:u + s * Ho:s, v:v + s * Wo:s] += cols[:, :, u, v]
    return x[:, :, p:p + H, p:p + W]

class Conv2d:
    def __init__(self, W, b, s=1, p=0):            # W: (Cout, Cin, k, k)
        self.params, self.s, self.p = {"W": W.copy(), "b": b.copy()}, s, p
    def forward(self, x):
        Cout, Cin, k, _ = self.params["W"].shape
        self.shape, self.k = x.shape, k
        self.cols, (Ho, Wo) = im2col(x, k, self.s, self.p)
        y = self.params["W"].reshape(Cout, -1) @ self.cols + self.params["b"][:, None]   # (B, Cout, Ho*Wo)
        return y.reshape(x.shape[0], Cout, Ho, Wo)
    def backward(self, g):                          # g: (B, Cout, Ho, Wo)
        B, Cout = g.shape[:2]
        g2 = g.reshape(B, Cout, -1)
        self.grads = {"W": np.einsum("bop,bcp->oc", g2, self.cols).reshape(self.params["W"].shape),
                      "b": g2.sum((0, 2))}
        dcols = np.einsum("oc,bop->bcp", self.params["W"].reshape(Cout, -1), g2)
        return col2im(dcols, self.shape, self.k, self.s, self.p)
# [[/im2col]]
xc = rng.normal(size=(2, 3, 7, 7)); Wc = rng.normal(size=(4, 3, 3, 3)); bc = rng.normal(size=4)
conv = Conv2d(Wc, bc, s=2, p=1)
yc = conv.forward(xc); gy = rng.normal(size=yc.shape); dx = conv.backward(gy)
xt_ = torch.tensor(xc, requires_grad=True); Wt_ = torch.tensor(Wc, requires_grad=True); bt_ = torch.tensor(bc, requires_grad=True)
yt_ = F.conv2d(xt_, Wt_, bt_, stride=2, padding=1); yt_.backward(torch.tensor(gy))
out.check("im2col conv forward vs F.conv2d", yc, yt_.detach().numpy())
out.check("conv backward dx vs autograd", dx, xt_.grad.numpy())
out.check("conv backward dW vs autograd", conv.grads["W"], Wt_.grad.numpy())
out.check("conv backward db vs autograd", conv.grads["b"], bt_.grad.numpy())
out.val("conv_out", "\\times".join(map(str, yc.shape)))


# ===================================================================== LSTM
sig = lambda z: 1 / (1 + np.exp(-z))
# [[lstm_layer]]
def lstm_forward(xs, Wih, Whh, bih, bhh):
    """xs: (T, B, D); weights in nn.LSTM layout (gates i, f, g, o). Returns all h_t."""
    T, B, _ = xs.shape
    H = Whh.shape[1]
    h, c = np.zeros((B, H)), np.zeros((B, H))
    hs = []
    for t in range(T):
        z = xs[t] @ Wih.T + h @ Whh.T + bih + bhh
        i, f, g, o = sig(z[:, :H]), sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sig(z[:, 3 * H:])
        c = f * c + i * g
        h = o * np.tanh(c)
        hs.append(h)
    return np.stack(hs), (h, c)
# [[/lstm_layer]]
torch.manual_seed(0)
tl = nn.LSTM(4, 6)
xs = rng.normal(size=(9, 3, 4))
hs, (hT, cT) = lstm_forward(xs, *(p.detach().numpy() for p in (tl.weight_ih_l0, tl.weight_hh_l0, tl.bias_ih_l0, tl.bias_hh_l0)))
ot, (htt, ctt) = tl(torch.tensor(xs))
out.check("NumPy LSTM over 9 steps vs nn.LSTM (outputs)", hs, ot.detach().numpy())
out.check("NumPy LSTM final cell state vs nn.LSTM", cT, ctt[0].detach().numpy())


# ============================================ character-level language model
text = ("a network learns a function from examples. it adjusts its weights to lower a loss. "
        "gradients flow backwards through every layer, and the optimiser takes a small step. "
        "repeat this many times and the loss falls, the weights settle, and the network predicts. ") * 3
chars = sorted(set(text)); stoi = {ch: i for i, ch in enumerate(chars)}; V = len(chars)
data = np.array([stoi[ch] for ch in text])
ctx, E_dim, Hn = 3, 8, 64
Xc_ = np.stack([data[i:i + ctx] for i in range(len(data) - ctx)]); Yc_ = data[ctx:]
# [[charlm]]
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
# [[/charlm]]
torch.manual_seed(0)
temb = nn.Embedding(V, E_dim); tmlp = nn.Sequential(nn.Linear(ctx * E_dim, Hn), nn.Tanh(), nn.Linear(Hn, V))
emb = Embedding(temb.weight.detach().numpy())
la = Linear(tmlp[0].weight.detach().numpy(), tmlp[0].bias.detach().numpy())
lb = Linear(tmlp[2].weight.detach().numpy(), tmlp[2].bias.detach().numpy())
body = Sequential(la, Tanh(), lb)
opt = Adam([emb, la, lb], lr=0.01)
topt = torch.optim.Adam(list(temb.parameters()) + list(tmlp.parameters()), lr=0.01)
g = np.random.default_rng(2)
ours_l, ref_l = [], []
for step in range(300):
    idx = g.choice(len(Yc_), 64, replace=False)
    h0 = emb.forward(Xc_[idx]); loss, gl = softmax_ce(body.forward(h0), Yc_[idx])
    emb.backward(body.backward(gl)); opt.step()
    topt.zero_grad()
    lt = F.cross_entropy(tmlp(temb(torch.tensor(Xc_[idx])).flatten(1)), torch.tensor(Yc_[idx])); lt.backward(); topt.step()
    ours_l.append(loss); ref_l.append(lt.item())
out.check("char-LM: NumPy training trajectory (300 Adam steps) vs PyTorch", ours_l, ref_l, atol=1e-8)
out.val("clm_V", V); out.val("clm_loss0", ours_l[0], 3); out.val("clm_loss_end", np.mean(ours_l[-20:]), 3)
out.val("clm_lnV", np.log(V), 3); out.val("clm_ppl", np.exp(np.mean(ours_l[-20:])), 2)
out.val("clm_params", sum(v.size for l in (emb, la, lb) for v in l.params.values()))
# greedy-ish sampling (temperature 0.5) from the NumPy model
gs = np.random.default_rng(3)
seq = [stoi[ch] for ch in "the"]
for _ in range(80):
    logits = body.forward(emb.forward(np.array([seq[-ctx:]])))[0] / 0.5
    p = np.exp(logits - logits.max()); p /= p.sum()
    seq.append(int(gs.choice(V, p=p)))
sample = "".join(chars[i] for i in seq)
import textwrap
out.text("sample", textwrap.fill(sample, 64))

# problem values
from common import thousands
out.val("p_cols", thousands(64 * 9 * 56 * 56)); out.val("p_in", thousands(64 * 56 * 56))
