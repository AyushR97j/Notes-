"""Chapter 3: backpropagation. A numeric 2-2-1 forward/backward pass, matrix-form
backprop for an L-layer MLP, gradient checking, a ~60-line reverse-mode autograd,
vanishing/exploding gradients through depth, dying ReLUs, FLOP counts."""
import numpy as np
import torch
import torch.nn.functional as F
from common import setup, thousands
from dlutil import sci

out = setup(__file__)
torch.set_default_dtype(torch.float64)

# -------------------------------------------------- tiny graph f = (x + y) * z
x, y, z = 2.0, -1.0, 3.0
q = x + y; f = q * z
out.val("g_q", q, 0); out.val("g_f", f, 0)
out.val("g_dfdx", z, 0); out.val("g_dfdz", q, 0)
xt, yt, zt = (torch.tensor(v, requires_grad=True) for v in (x, y, z))
((xt + yt) * zt).backward()
out.check("graph grads vs autograd", [z, z, q], [xt.grad.item(), yt.grad.item(), zt.grad.item()])

# ------------------------------------------------------ the 2-2-1 network
sig = lambda t: 1 / (1 + np.exp(-t))
x0 = np.array([1.0, 0.5]); y0 = 1.0
W1 = np.array([[0.2, -0.3], [0.4, 0.1]]); b1 = np.array([0.1, -0.1])
W2 = np.array([0.5, -0.6]); b2 = 0.05
eta = 0.5
# [[net221]]
# forward
z1 = W1 @ x0 + b1            # (2,)
h1 = sig(z1)                 # hidden activations
z2 = W2 @ h1 + b2            # scalar logit
yhat = sig(z2)
L = -(y0 * np.log(yhat) + (1 - y0) * np.log(1 - yhat))       # BCE
# backward
d2 = yhat - y0               # dL/dz2  (sigmoid + BCE)
gW2 = d2 * h1; gb2 = d2
d1 = (W2 * d2) * h1 * (1 - h1)                                # dL/dz1
gW1 = np.outer(d1, x0); gb1 = d1
# one SGD step
W1n, b1n, W2n, b2n = W1 - eta * gW1, b1 - eta * gb1, W2 - eta * gW2, b2 - eta * gb2
# [[/net221]]
P = [torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, np.array(b2))]
zz1 = P[0] @ torch.tensor(x0) + P[1]
yy = torch.sigmoid(P[2] @ torch.sigmoid(zz1) + P[3])
Lt = F.binary_cross_entropy(yy, torch.tensor(y0))
Lt.backward()
out.check("2-2-1 loss", L, Lt.item())
out.check("2-2-1 dW1", gW1, P[0].grad.numpy()); out.check("2-2-1 db1", gb1, P[1].grad.numpy())
out.check("2-2-1 dW2", gW2, P[2].grad.numpy()); out.check("2-2-1 db2", gb2, P[3].grad.item())
for i in range(2):
    out.val(f"z1_{i}", z1[i], 4); out.val(f"h1_{i}", h1[i], 4)
    out.val(f"d1_{i}", d1[i], 4); out.val(f"gW2_{i}", gW2[i], 4); out.val(f"gb1_{i}", gb1[i], 4)
    out.val(f"W2n_{i}", W2n[i], 4); out.val(f"b1n_{i}", b1n[i], 4)
    for j in range(2):
        out.val(f"gW1_{i}{j}", gW1[i, j], 4); out.val(f"W1n_{i}{j}", W1n[i, j], 4)
out.val("z2", z2, 4); out.val("yhat", yhat, 4); out.val("L", L, 4); out.val("d2", d2, 4)
out.val("b2n", b2n, 4)
Ln = -np.log(sig(W2n @ sig(W1n @ x0 + b1n) + b2n))
out.val("Lnew", Ln, 4)
out.val("w2d2_0", W2[0] * d2, 4); out.val("w2d2_1", W2[1] * d2, 4)
out.val("sp_0", h1[0] * (1 - h1[0]), 4); out.val("sp_1", h1[1] * (1 - h1[1]), 4)

# ----------------------------------- matrix-form backprop for an L-layer MLP
# [[mlp_backprop]]
def mlp_forward_backward(X, y, Ws, bs):
    """ReLU hidden layers, softmax + cross-entropy output. X: (B, n0), y: (B,) ints.
    Ws[l]: (n_{l+1}, n_l) as in nn.Linear.  Returns loss and gradients."""
    B = X.shape[0]
    hs, zs = [X], []
    for l, (W, b) in enumerate(zip(Ws, bs)):
        z = hs[-1] @ W.T + b                      # (B, n_{l+1})
        zs.append(z)
        hs.append(np.maximum(z, 0) if l < len(Ws) - 1 else z)
    Z = zs[-1] - zs[-1].max(1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(B), y]))
    delta = P.copy(); delta[np.arange(B), y] -= 1; delta /= B      # dL/dz_L
    gWs, gbs = [None] * len(Ws), [None] * len(Ws)
    for l in reversed(range(len(Ws))):
        gWs[l] = delta.T @ hs[l]                  # (n_{l+1}, n_l)
        gbs[l] = delta.sum(0)
        if l > 0:
            delta = (delta @ Ws[l]) * (zs[l - 1] > 0)  # back through W, then ReLU
    return loss, gWs, gbs
# [[/mlp_backprop]]
rng = np.random.default_rng(0)
sizes = [5, 8, 6, 3]
Ws = [rng.normal(size=(m, n)) / np.sqrt(n) for n, m in zip(sizes[:-1], sizes[1:])]
bs = [rng.normal(size=m) * 0.1 for m in sizes[1:]]
Xb = rng.normal(size=(16, 5)); yb = rng.integers(0, 3, 16)
loss, gWs, gbs = mlp_forward_backward(Xb, yb, Ws, bs)
Wt = [torch.tensor(W, requires_grad=True) for W in Ws]; bt = [torch.tensor(b, requires_grad=True) for b in bs]
h = torch.tensor(Xb)
for l in range(3):
    h = h @ Wt[l].T + bt[l]
    if l < 2:
        h = torch.relu(h)
lt = F.cross_entropy(h, torch.tensor(yb)); lt.backward()
out.check("MLP loss vs torch", loss, lt.item())
for l in range(3):
    out.check(f"MLP dW{l+1} vs autograd", gWs[l], Wt[l].grad.numpy())
    out.check(f"MLP db{l+1} vs autograd", gbs[l], bt[l].grad.numpy())

# --------------------------------------------------------- gradient checking
# [[gradcheck]]
def numerical_grad(f, theta, h=1e-5):
    g = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta); e.flat[i] = h
        g.flat[i] = (f(theta + e) - f(theta - e)) / (2 * h)      # centred difference
    return g

def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b)))
# [[/gradcheck]]
def loss_W1(Wflat):
    W = Ws.copy(); W[0] = Wflat.reshape(Ws[0].shape)
    return mlp_forward_backward(Xb, yb, W, bs)[0]
gnum = numerical_grad(loss_W1, Ws[0].ravel()).reshape(Ws[0].shape)
re = rel_error(gnum, gWs[0])
out.val("gc_relerr", sci(re, 1))
out.check("numerical vs analytic dW1", gnum, gWs[0], atol=1e-8)
# error vs step size: forward vs centred difference on one weight
hs_ = 10.0 ** np.arange(-1, -13, -0.5)
i0 = (0, 0)
def f1(t):
    W = [w.copy() for w in Ws]; W[0][i0] = t
    return mlp_forward_backward(Xb, yb, W, bs)[0]
t0 = Ws[0][i0]; ga = gWs[0][i0]
fwd = [abs((f1(t0 + h) - f1(t0)) / h - ga) / abs(ga) for h in hs_]
cen = [abs((f1(t0 + h) - f1(t0 - h)) / (2 * h) - ga) / abs(ga) for h in hs_]
out.dat("gradcheck", {"h": hs_, "fwd": fwd, "cen": cen}, nd=16)
out.val("gc_best_cen", sci(min(cen), 0)); out.val("gc_best_fwd", sci(min(fwd), 0))
out.val("gc_best_h_cen", sci(hs_[int(np.argmin(cen))], 0)); out.val("gc_best_h_fwd", sci(hs_[int(np.argmin(fwd))], 0))
# a planted bug: forgetting the ReLU mask in the backward pass
def buggy(X, y, Ws, bs):
    B = X.shape[0]; hs, zs = [X], []
    for l, (W, b) in enumerate(zip(Ws, bs)):
        z = hs[-1] @ W.T + b; zs.append(z); hs.append(np.maximum(z, 0) if l < len(Ws) - 1 else z)
    Z = zs[-1] - zs[-1].max(1, keepdims=True); Pm = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    delta = Pm.copy(); delta[np.arange(B), y] -= 1; delta /= B
    g = None
    for l in reversed(range(len(Ws))):
        g = delta.T @ hs[l]
        if l > 0:
            delta = delta @ Ws[l]                 # BUG: no ReLU mask
    return g
out.val("gc_bug_relerr", rel_error(gnum, buggy(Xb, yb, Ws, bs)), 2)

# ---------------------------------------------- reverse-mode autograd, ~60 lines
# [[autograd]]
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
# [[/autograd]]
# check on a 2-layer tanh/ReLU net with a re-used weight (fan-out) and broadcasting
A = [Tensor(Ws[0].T), Tensor(bs[0][None, :]), Tensor(Ws[1].T), Tensor(bs[1][None, :]),
     Tensor(Ws[2].T), Tensor(bs[2][None, :])]
Xa = Tensor(Xb)
hA = (Xa @ A[0] + A[1]).relu()
hB = (hA @ A[2] + A[3]).tanh()
logits = hB @ A[4] + A[5] + (hB * hB).sum() * Tensor(0.1)      # scalar broadcast
lossA = logits.cross_entropy(yb)
lossA.backward()
T = [torch.tensor(a.data, requires_grad=True) for a in A]
hh = torch.relu(torch.tensor(Xb) @ T[0] + T[1])
hh2 = torch.tanh(hh @ T[2] + T[3])
lg = hh2 @ T[4] + T[5] + (hh2 * hh2).sum() * 0.1
lt = F.cross_entropy(lg, torch.tensor(yb)); lt.backward()
out.check("mini-autograd loss", lossA.data, lt.item())
for k in range(6):
    out.check(f"mini-autograd grad {k}", A[k].grad, T[k].grad.numpy())
src = open(__file__).read()
a = src.index("# [[autograd]]"); b = src.index("# [[/autograd]]")
out.val("autograd_lines", len([l for l in src[a:b].splitlines()[1:] if l.strip()]))

# ---------------------------------------------- vanishing / exploding gradients
def grad_profile(act, std_fn, depth=30, width=64, B=128, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(B, width, requires_grad=True)
    hs = []
    for l in range(depth):
        W = torch.randn(width, width) * std_fn(width)
        h = act(h @ W)
        h.retain_grad(); hs.append(h)
    (h * torch.randn_like(h)).sum().backward()
    return np.array([hh.grad.norm().item() for hh in hs])
cfg = {
    "sig": (torch.sigmoid, lambda n: (1 / n) ** 0.5),
    "tanh": (torch.tanh, lambda n: (1 / n) ** 0.5),
    "relu": (torch.relu, lambda n: (2 / n) ** 0.5),
    "relubig": (torch.relu, lambda n: 1.5 * (2 / n) ** 0.5),
}
prof = {k: grad_profile(*v) for k, v in cfg.items()}
cols = {"layer": np.arange(1, 31)}
for k, v in prof.items():
    cols[k] = np.log10(v / v[-1])          # relative to the last layer
    out.val(f"vg_ratio_{k}", sci(v[0] / v[-1], 1))
out.dat("vanish", cols)

# ------------------------------------------------------------------ dying ReLU
def dead_fraction(lr, steps=300, seed=0):
    torch.manual_seed(seed)
    Xr = torch.randn(512, 10)
    yr = (Xr[:, :3].sum(1, keepdim=True)) ** 2
    net = torch.nn.Sequential(torch.nn.Linear(10, 200), torch.nn.ReLU(), torch.nn.Linear(200, 1))
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(); F.mse_loss(net(Xr), yr).backward()
        opt.step()
    with torch.no_grad():
        Hd = torch.relu(net[0](torch.randn(4096, 10)))
        final = F.mse_loss(net(Xr), yr).item()
    return (Hd.max(0).values == 0).float().mean().item(), final
for lr in (0.01, 0.1, 0.2):
    fr, fl = dead_fraction(lr)
    out.val(f"dead_{str(lr).replace('.', '')}", 100 * fr, 1)
    out.val(f"deadloss_{str(lr).replace('.', '')}", fl, 2)
# dead from a bad bias init
torch.manual_seed(0)
lin = torch.nn.Linear(10, 200)
with torch.no_grad():
    lin.bias.fill_(-3.0)
Hb = torch.relu(lin(torch.randn(4096, 10)))
out.val("dead_bias", 100 * (Hb.max(0).values == 0).float().mean().item(), 1)

# ------------------------------------------------------------------ FLOPs
sizes = [784, 256, 128, 10]
fwd = 2 * sum(m * n for m, n in zip(sizes[:-1], sizes[1:]))
out.val("flops_fwd", thousands(fwd)); out.val("flops_train", thousands(3 * fwd))
macs = sum(m * n for m, n in zip(sizes[:-1], sizes[1:]))
out.val("macs_fwd", thousands(macs))

# MSE variant of the 2-2-1 example: extra factor yhat (1 - yhat)
out.val("mse_factor", yhat * (1 - yhat), 4)
out.val("mse_d2", d2 * yhat * (1 - yhat), 4)
