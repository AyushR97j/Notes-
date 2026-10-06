"""Chapter 1: neuron outputs, the perceptron algorithm (trace, mistake bound,
XOR failure), a hand-built XOR network, parameter counting, linear collapse,
universal approximation with ReLU hinges, depth vs width, capacity."""
import numpy as np
import torch
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from common import setup, thousands

out = setup(__file__)

# ---------------------------------------------------------------- a neuron
# [[neuron]]
x = np.array([0.5, -1.0, 2.0])
w = np.array([0.4, 0.3, -0.2])
b = 0.1
z = w @ x + b                                   # weighted sum (pre-activation)
acts = {
    "step":    float(z >= 0),
    "sigmoid": 1 / (1 + np.exp(-z)),
    "tanh":    np.tanh(z),
    "relu":    max(0.0, z),
}
# [[/neuron]]
zt = torch.tensor(x) @ torch.tensor(w) + b
out.check("neuron z", z, zt.item())
out.check("neuron sigmoid", acts["sigmoid"], torch.sigmoid(zt).item())
out.check("neuron tanh", acts["tanh"], torch.tanh(zt).item())
out.check("neuron relu", acts["relu"], torch.relu(zt).item())
out.val("neuron_z", z, 2)
for k, v in acts.items():
    out.val(f"neuron_{k}", v, 4)
# a second neuron used in the OA-style question: 4 inputs, sigmoid
x2 = np.array([1.0, 0.0, 1.0, 1.0]); w2 = np.array([0.6, -0.9, -0.4, 0.5]); b2 = -0.3
z2 = w2 @ x2 + b2
out.val("q_z", z2, 2); out.val("q_sig", 1 / (1 + np.exp(-z2)), 4)

# ------------------------------------------------------------- perceptron
X = np.array([[3.0, 1.0], [1.0, -2.0], [3.0, 2.0], [-2.0, -3.0], [2.0, -2.0], [-3.0, 1.0]])
y = np.array([1, -1, 1, -1, 1, -1])

# [[perceptron]]
def perceptron(X, y, epochs=20, lr=1.0):
    w, b = np.zeros(X.shape[1]), 0.0
    log = []
    for ep in range(epochs):
        mistakes = 0
        for i in range(len(X)):
            if y[i] * (w @ X[i] + b) <= 0:      # mistake (or on the boundary)
                w, b = w + lr * y[i] * X[i], b + lr * y[i]
                mistakes += 1
                log.append((ep + 1, i + 1, w.copy(), b))
        if mistakes == 0:
            return w, b, log, ep + 1
    return w, b, log, epochs
# [[/perceptron]]

wp, bp, log, eps = perceptron(X, y)
ref = Perceptron(eta0=1.0, shuffle=False, tol=None, max_iter=eps, penalty=None,
                 fit_intercept=True).fit(X, y)
out.check("perceptron weights vs sklearn", np.r_[wp, bp], np.r_[ref.coef_.ravel(), ref.intercept_])
rows = []
for ep, i, wv, bv in log:
    rows.append(f"{ep} & $\\x_{{{i}}}$ & $({wv[0]:g},\\ {wv[1]:g})$ & ${bv:g}$ \\\\")
out.tex("trace_rows", "\n".join(rows))
out.tex("data_list", ", ".join(
    f"$\\x_{{{i+1}}}=({X[i,0]:g},{X[i,1]:g})^{{{'+' if y[i] > 0 else '-'}}}$" for i in range(len(X))))
out.val("n_updates", len(log)); out.val("n_epochs", eps)
out.val("w1", wp[0], 0); out.val("w2", wp[1], 0); out.val("bfin", bp, 0)
# learning-rate invariance with zero init: lr=0.1 gives the same boundary
w01, b01, log01, _ = perceptron(X, y, lr=0.1)
out.check("lr invariance (w scaled by 0.1)", np.r_[w01, b01], 0.1 * np.r_[wp, bp])

# mistake bound (R/gamma)^2 using the max-margin separator as w*
Xa = np.c_[X, np.ones(len(X))]                      # augmented inputs
svm = SVC(kernel="linear", C=1e6).fit(X, y)
ws = np.r_[svm.coef_.ravel(), svm.intercept_]
ws = ws / np.linalg.norm(ws)
gamma = np.min(y * (Xa @ ws))
R = np.max(np.linalg.norm(Xa, axis=1))
out.val("R", R, 3); out.val("gamma", gamma, 3); out.val("bound", (R / gamma) ** 2, 1)
out.dat("pos", {"x1": X[y > 0, 0], "x2": X[y > 0, 1]})
out.dat("neg", {"x1": X[y < 0, 0], "x2": X[y < 0, 1]})
xs = np.linspace(-3.5, 3.5, 2)
out.dat("line", {"x": xs, "y": -(wp[0] * xs + bp) / wp[1]})

# --------------------------------------------------------------------- XOR
Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
yx = np.array([-1, 1, 1, -1])
wx, bx, logx, epx = perceptron(Xx, yx, epochs=50)
per_epoch = np.bincount([e for e, *_ in logx], minlength=51)[1:]
out.val("xor_epochs", 50); out.val("xor_min_mistakes", int(per_epoch.min()))
out.val("xor_total", len(logx))
skl = Perceptron(eta0=1.0, shuffle=False, tol=None, max_iter=50, penalty=None).fit(Xx, yx)
out.val("xor_sklearn_acc", skl.score(Xx, yx), 2)

# [[xor_hand]]
step = lambda t: (t >= 0).astype(float)
def xor_net(X):
    h1 = step(X @ np.array([1, 1]) - 0.5)     # OR
    h2 = step(X @ np.array([1, 1]) - 1.5)     # AND
    return step(h1 - h2 - 0.5), h1, h2        # OR and not AND
# [[/xor_hand]]
yo, h1, h2 = xor_net(Xx)
out.check("hand XOR net", yo, (yx > 0).astype(float))
out.tex("xor_rows", "\n".join(
    f"{int(a)} & {int(c)} & {int(p)} & {int(q)} & {int(o)} \\\\" for (a, c), p, q, o in zip(Xx, h1, h2, yo)))

# trained 2-H-1 tanh nets on XOR, 20 seeds trained in one batched tensor
def batched_mlp_train(X, y, S, H, steps, lr, hidden_mask=None, p_drop=0.0, seed=0):
    """S independent 1-hidden-layer nets (nn.Linear-style init) trained with Adam
    on BCE; the summed loss gives each net its own gradient."""
    g = torch.Generator().manual_seed(seed)
    d = X.shape[1]
    u = lambda shape, fan: (torch.rand(shape, generator=g) * 2 - 1) / fan ** 0.5
    P = [u((S, d, H), d), u((S, 1, H), d), u((S, H, 1), H), u((S, 1, 1), H)]
    for t in P:
        t.requires_grad_(True)
    opt = torch.optim.Adam(P, lr=lr)
    mask = torch.ones(S, 1, H) if hidden_mask is None else hidden_mask
    def fwd(train):
        h = torch.tanh(X @ P[0] + P[1]) if H <= 8 else torch.relu(X @ P[0] + P[1])
        h = h * mask
        if train and p_drop > 0:
            keep = (torch.rand(h.shape, generator=g) > p_drop).float()
            h = h * keep / (1 - p_drop)
        return h @ P[2] + P[3]
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fwd(True), y.expand(S, -1, -1), reduction="none").mean(dim=(1, 2)).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        return ((fwd(False) > 0).float() == y).float().mean(dim=(1, 2))
Xt = torch.tensor(Xx, dtype=torch.float32); yt = torch.tensor(yx > 0, dtype=torch.float32)[:, None]
acc2 = batched_mlp_train(Xt, yt, 20, 2, 1500, 0.05, seed=0)
acc8 = batched_mlp_train(Xt, yt, 20, 8, 1500, 0.05, seed=0)
out.val("xor_seeds", 20)
out.val("xor_solved2", int((acc2 == 1).sum())); out.val("xor_solved8", int((acc8 == 1).sum()))

# ------------------------------------------------------- parameter counting
# [[count]]
def mlp_params(sizes):
    return sum(m * n + n for m, n in zip(sizes[:-1], sizes[1:]))
sizes = [784, 256, 128, 10]
n_ours = mlp_params(sizes)
# [[/count]]
layers = []
for m, n in zip(sizes[:-1], sizes[1:]):
    layers += [torch.nn.Linear(m, n), torch.nn.ReLU()]
mlp = torch.nn.Sequential(*layers[:-1])
n_torch = sum(p.numel() for p in mlp.parameters())
out.check("MLP 784-256-128-10 param count", n_ours, n_torch, atol=0)
out.val("mlp_params", thousands(n_ours))
for j, (m, n) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
    out.val(f"mlp_l{j}", thousands(m * n + n))
# OA one-liner: 3 inputs, hidden 4, output 2
out.val("small_params", mlp_params([3, 4, 2]))
out.val("p1a", mlp_params([100, 50, 50, 3]))
out.val("p1a_l1", mlp_params([100, 50])); out.val("p1a_l2", mlp_params([50, 50])); out.val("p1a_l3", mlp_params([50, 3]))

# linear layers collapse without a non-linearity
rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4)
W2, b2 = rng.normal(size=(2, 4)), rng.normal(size=2)
xin = rng.normal(size=3)
two_layer = W2 @ (W1 @ xin + b1) + b2
one_layer = (W2 @ W1) @ xin + (W2 @ b1 + b2)
out.check("two linear layers == one linear layer", two_layer, one_layer)

# --------------------------------------------- universal approximation (1-D)
f = lambda t: np.sin(2 * np.pi * t) + 0.5 * t
# [[relu_interp]]
def relu_net_interp(f, n):
    """One hidden ReLU layer with n units that interpolates f at n+1 knots."""
    knots = np.linspace(0, 1, n + 1)
    vals = f(knots)
    slopes = np.diff(vals) / np.diff(knots)
    a = np.r_[slopes[0], np.diff(slopes)]           # output weights
    def net(t):
        H = np.maximum(0.0, t[:, None] - knots[:-1][None, :])   # hidden layer
        return vals[0] + H @ a
    return net
# [[/relu_interp]]
tt = np.linspace(0, 1, 401)
errs = []
cols = {"t": tt, "f": f(tt)}
for n in [2, 4, 8, 16, 32, 64]:
    net = relu_net_interp(f, n)
    knots = np.linspace(0, 1, n + 1)
    out.check(f"ReLU net == piecewise-linear interp (n={n})", net(tt), np.interp(tt, knots, f(knots)))
    e = np.max(np.abs(net(tt) - f(tt)))
    errs.append(e)
    out.val(f"uat_err{n}", e, 4)
    if n in (4, 8):
        cols[f"n{n}"] = net(tt)
out.dat("uat", cols)
out.dat("uat_err", {"n": [2, 4, 8, 16, 32, 64], "err": errs})
out.val("uat_ratio", errs[-2] / errs[-1], 2)
out.val("uat_bound64", (1 / 64) ** 2 * 4 * np.pi ** 2 / 8, 4)   # h^2 max|f''| / 8

# ------------------------------------------------ depth: sawtooth via tent maps
# [[tent]]
def tent_layer(t):            # 2 ReLU units: g(t) = 2 relu(t) - 4 relu(t - 1/2)
    return 2 * np.maximum(0, t) - 4 * np.maximum(0, t - 0.5)
def deep_tent(t, L):
    for _ in range(L):
        t = tent_layer(t)
    return t
# [[/tent]]
tt2 = np.linspace(0, 1, 2 ** 12 + 1)
pieces = {}
for L in range(1, 7):
    yv = deep_tent(tt2, L)
    s = np.round(np.diff(yv) / np.diff(tt2), 6)
    pieces[L] = 1 + int(np.sum(s[1:] != s[:-1]))
    out.val(f"pieces{L}", pieces[L])
    assert pieces[L] == 2 ** L
out.val("pieces_units6", 2 * 6)
dd = {"t": tt2[::16]}
for L in (1, 2, 3):
    dd[f"L{L}"] = deep_tent(tt2[::16], L)
out.dat("tent", dd)

# --------------------------------------------------------- capacity vs width
# random labels: the only way to fit them is to memorise; widths share one
# batched tensor (unused hidden units are masked to zero and never move)
n, d = 200, 10
Xc = torch.randn(n, d)
yc = torch.randint(0, 2, (n,)).float()[:, None]
widths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
mask = torch.zeros(len(widths), 1, 256)
for i, wd in enumerate(widths):
    mask[i, 0, :wd] = 1
acc = batched_mlp_train(Xc, yc, len(widths), 256, 600, 0.01, hidden_mask=mask, seed=1)
accd = batched_mlp_train(Xc, yc, len(widths), 256, 600, 0.01, hidden_mask=mask, p_drop=0.5, seed=1)
out.dat("capacity", {"width": widths, "acc": acc.numpy(), "accdrop": accd.numpy()})
out.val("cap_acc1", 100 * acc[0].item(), 1); out.val("cap_acc256", 100 * acc[-1].item(), 1)
out.val("cap_drop256", 100 * accd[-1].item(), 1); out.val("cap_drop16", 100 * accd[4].item(), 1)
out.val("cap_acc16", 100 * acc[4].item(), 1)
out.val("cap_n", n); out.val("cap_d", d)
