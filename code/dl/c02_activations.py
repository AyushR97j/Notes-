"""Chapter 2: activation functions and their derivatives, stable softmax and
log-sum-exp, cross-entropy from logits (many OA-shaped examples), temperature,
BCE with logits, MSE vs CE gradients, hinge/Huber/focal/label smoothing."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp as sp_lse, softmax as sp_softmax
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from common import setup
from dlutil import sci

out = setup(__file__)
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------- activations
# [[acts]]
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
# [[/acts]]
zs = np.linspace(-4, 4, 161)
ours = {"sigmoid": sigmoid, "tanh": np.tanh, "relu": relu, "leaky": lambda z: leaky(z, 0.1),
        "elu": elu, "gelu": gelu, "silu": silu, "softplus": softplus}
refs = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu,
        "leaky": lambda t: F.leaky_relu(t, 0.1), "elu": F.elu, "gelu": F.gelu,
        "silu": F.silu, "softplus": F.softplus}
dours = {"sigmoid": d_sigmoid, "tanh": d_tanh, "relu": d_relu, "gelu": d_gelu, "silu": d_silu}
zt = torch.tensor(zs, requires_grad=True)
cols, dcols = {"z": zs}, {"z": zs}
for k in ours:
    yt = refs[k](zt)
    out.check(f"{k} vs torch", ours[k](zs), yt.detach().numpy())
    cols[k] = ours[k](zs)
    if k in dours:
        g, = torch.autograd.grad(yt.sum(), zt)
        mask = np.abs(zs) > 1e-9          # skip the kink of ReLU at 0
        out.check(f"d {k} vs autograd", dours[k](zs)[mask], g.numpy()[mask])
        dcols[k] = dours[k](zs)
out.check("gelu tanh-approx vs torch approximate='tanh'", gelu_tanh(zs),
          F.gelu(torch.tensor(zs), approximate="tanh").numpy())
out.dat("acts", cols)
out.dat("dacts", dcols)
out.val("gelu_approx_err", np.max(np.abs(gelu(np.linspace(-6, 6, 2001)) - gelu_tanh(np.linspace(-6, 6, 2001)))), 5)
out.val("dsig_max", d_sigmoid(0.0), 2)
out.val("dsig_5", d_sigmoid(5.0), 4)
out.val("dsig_pow10", sci(0.25 ** 10))
res = minimize_scalar(silu, bounds=(-5, 0), method="bounded")
out.val("silu_argmin", res.x, 3); out.val("silu_min", res.fun, 3)
res = minimize_scalar(gelu, bounds=(-5, 0), method="bounded")
out.val("gelu_argmin", res.x, 3); out.val("gelu_min", res.fun, 3)
# value table at a few points (OA: "compute f(-2), f(0), f(2)")
rows = []
for name, f in [("sigmoid", sigmoid), ("tanh", np.tanh), ("ReLU", relu),
                ("leaky ReLU ($0.01$)", leaky), ("ELU ($\\alpha=1$)", elu),
                ("GELU", gelu), ("SiLU", silu)]:
    v = [f(np.array(t)) for t in (-2.0, -0.5, 0.0, 0.5, 2.0)]
    rows.append(name + " & " + " & ".join(f"${float(a):.4f}$" for a in v) + " \\\\")
out.tex("act_table", "\n".join(rows))

# ------------------------------------------------------------- softmax / LSE
# [[softmax]]
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)        # shift: same result, no overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def logsumexp(z):
    m = z.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True))).squeeze(-1)

def cross_entropy(z, y):                         # z: (B, K) logits, y: (B,) ints
    return np.mean(logsumexp(z) - z[np.arange(len(y)), y])
# [[/softmax]]
big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
out.val("naive_nan", "NaN" if np.isnan(naive).all() else "ok")
sm = softmax(big)
out.check("stable softmax vs torch", sm, torch.softmax(torch.tensor(big), 0).numpy())
out.check("logsumexp vs scipy", logsumexp(big), sp_lse(big))
for i in range(3):
    out.val(f"big_p{i}", sm[i], 3)
out.val("lse_big", logsumexp(big), 4)

# the five-logit OA question
z5 = np.array([2.0, 1.0, 0.1, -1.0, 0.5])
p5 = softmax(z5)
out.check("softmax(z5) vs scipy", p5, sp_softmax(z5))
ce5 = -np.log(p5[0])
out.check("CE(z5, class 0) vs torch", ce5, F.cross_entropy(torch.tensor(z5)[None], torch.tensor([0])).item())
for i in range(5):
    out.val(f"z5_e{i}", np.exp(z5[i]), 4)
    out.val(f"z5_p{i}", p5[i], 4)
out.val("z5_sum", np.exp(z5).sum(), 4)
out.val("z5_ce0", ce5, 4)
out.val("z5_ce3", -np.log(p5[3]), 4)
out.val("z5_conf", p5.max() * 100, 1)

# many examples (table)
rng = np.random.default_rng(2)
ex_logits = [np.array(v) for v in ([3.0, 1.0, 0.2], [0.0, 0.0, 0.0, 0.0], [2.0, 2.0, -1.0],
                                   [5.0, -2.0, 0.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0],
                                   [-1.0, 0.5, 0.5, 2.5, 0.0], [10.0, 0.0])]
ex_true = [0, 2, 1, 1, 0, 3, 0]
rows = []
for z, y in zip(ex_logits, ex_true):
    p = softmax(z)
    ce = -np.log(p[y])
    ref = F.cross_entropy(torch.tensor(z)[None], torch.tensor([y])).item()
    out.check(f"CE example {z.tolist()} y={y}", ce, ref)
    zs_ = "(" + ",\\,".join(f"{v:g}" for v in z) + ")"
    ps_ = "(" + ",\\,".join(f"{v:.3f}" for v in p) + ")"
    rows.append(f"${zs_}$ & {y} & ${ps_}$ & {int(np.argmax(p))} & ${p[y]:.3f}$ & ${ce:.3f}$ \\\\")
out.tex("ce_table", "\n".join(rows))
B, K = 64, 10
zb = rng.normal(size=(B, K)) * 0.01; yb = rng.integers(0, K, B)
ceb = cross_entropy(zb, yb)
out.check("batched CE vs torch", ceb, F.cross_entropy(torch.tensor(zb), torch.tensor(yb)).item())
out.val("ce_init", ceb, 4); out.val("ln10", np.log(10), 4)

# shift and scale
out.check("softmax shift invariance", softmax(z5 + 100), softmax(z5))
rows = []
for T in (0.5, 1.0, 2.0, 5.0):
    p = softmax(z5 / T)
    rows.append(f"${T:g}$ & " + " & ".join(f"${v:.3f}$" for v in p) +
                f" & ${-np.sum(p * np.log(p)):.3f}$ \\\\")
out.tex("temp_table", "\n".join(rows))
out.val("lnfive", np.log(5), 3)
T_cols = {"k": np.arange(1, 6)}
for T in (0.5, 1.0, 2.0):
    T_cols[f"T{str(T).replace('.', '')}"] = softmax(z5 / T)
out.dat("temp", T_cols)

# sigmoid = 2-class softmax
a, bq = 1.3, -0.4
out.check("softmax([a,b])[0] == sigmoid(a-b)", softmax(np.array([a, bq]))[0], sigmoid(a - bq))

# softmax Jacobian diag(p) - p p^T
J = np.diag(p5) - np.outer(p5, p5)
Jt = torch.autograd.functional.jacobian(lambda t: torch.softmax(t, 0), torch.tensor(z5))
out.check("softmax Jacobian vs autograd", J, Jt.numpy())
# gradient of CE wrt logits = p - y
zt5 = torch.tensor(z5, requires_grad=True)
F.cross_entropy(zt5[None], torch.tensor([0])).backward()
g = p5.copy(); g[0] -= 1
out.check("dCE/dz = p - onehot", g, zt5.grad.numpy())
for i in range(5):
    out.val(f"z5_g{i}", g[i], 4)

# ------------------------------------------------------------- binary losses
def bce(p, y): return -(y * np.log(p) + (1 - y) * np.log(1 - p))
for nm, p, y in [("bce_09", 0.9, 1), ("bce_01", 0.1, 1), ("bce_05", 0.5, 1), ("bce_099", 0.99, 0)]:
    v = bce(p, y)
    out.check(nm, v, F.binary_cross_entropy(torch.tensor([p]), torch.tensor([float(y)])).item())
    out.val(nm, v, 4)
# [[bce_logits]]
def bce_with_logits(z, y):          # = -[y log s(z) + (1-y) log(1-s(z))], stable
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
# [[/bce_logits]]
zz = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0]); yy = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
out.check("BCE with logits vs torch", bce_with_logits(zz, yy),
          F.binary_cross_entropy_with_logits(torch.tensor(zz), torch.tensor(yy), reduction="none").numpy())
with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    naive = bce(sigmoid(zz), yy)
assert np.isinf(naive[0]) and np.isinf(naive[4])
out.val("naive_bce_m1000", "\\infty")
out.val("stable_bce_m1000", bce_with_logits(zz, yy)[0], 1)

# MSE vs CE gradient through a sigmoid when y = 1
zg = np.linspace(-8, 8, 161)
s = sigmoid(zg)
g_ce = s - 1
g_mse = (s - 1) * s * (1 - s)        # d/dz of 0.5 (s - y)^2
zgt = torch.tensor(zg, requires_grad=True)
(0.5 * (torch.sigmoid(zgt) - 1) ** 2).sum().backward()
out.check("MSE-through-sigmoid gradient vs autograd", g_mse, zgt.grad.numpy())
out.dat("msegrad", {"z": zg, "ce": np.abs(g_ce), "mse": np.abs(g_mse)})
out.val("gce_m5", abs(sigmoid(-5) - 1), 4); out.val("gmse_m5", abs((sigmoid(-5) - 1) * d_sigmoid(-5)), 4)

# losses as functions of the margin m = y z (y in {-1, +1})
m = np.linspace(-2.5, 2.5, 201)
out.dat("margin", {"m": m, "zeroone": (m <= 0).astype(float), "hinge": np.maximum(0, 1 - m),
                   "logistic": np.log2(1 + np.exp(-m)), "square": (1 - m) ** 2})

# Huber
# [[huber]]
def huber(r, delta=1.0):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))
# [[/huber]]
r = np.array([-3.0, -1.0, -0.5, 0.0, 0.2, 2.0, 10.0])
out.check("Huber vs torch", huber(r), F.huber_loss(torch.tensor(r), torch.zeros(7), reduction="none").numpy())
out.val("huber_10", huber(np.array(10.0)), 1); out.val("mse_10", 0.5 * 100, 1)
out.val("huber_02", huber(np.array(0.2)), 2)

# focal loss
# [[focal]]
def focal(p_true, gamma=2.0, alpha=1.0):
    return -alpha * (1 - p_true) ** gamma * np.log(p_true)
# [[/focal]]
out.check("focal with gamma=0 is CE", focal(p5[0], 0.0), ce5)
for pt in (0.9, 0.5, 0.1):
    out.val(f"focal_{int(pt*10)}", focal(pt), 4)
    out.val(f"ce_{int(pt*10)}", -np.log(pt), 4)
    out.val(f"focal_ratio_{int(pt*10)}", (1 - pt) ** 2, 2)

# label smoothing
eps, Kc = 0.1, 5
q = np.full(Kc, eps / Kc); q[0] += 1 - eps
ls = -np.sum(q * np.log(p5))
out.check("label smoothing CE vs torch", ls,
          F.cross_entropy(torch.tensor(z5)[None], torch.tensor([0]), label_smoothing=eps).item())
out.val("ls_target_true", q[0], 2); out.val("ls_target_other", q[1], 2)
out.val("ls_ce", ls, 4)

# hinge (multiclass, Weston-Watkins form) on z5 with true class 1
mh = np.sum(np.maximum(0, np.delete(z5, 1) - z5[1] + 1))
out.check("multiclass hinge vs torch multi_margin_loss*(K)",
          mh / Kc, F.multi_margin_loss(torch.tensor(z5)[None], torch.tensor([1])).item())
out.val("hinge5", mh, 2)

# problem: logit increase that halves the loss at p_y = 0.2
pp = 0.2; pn = np.sqrt(pp)
delta = np.log((pn / (1 - pn)) / (pp / (1 - pp)))
zc = np.array([np.log(pp / (1 - pp)), 0.0])           # two-class check
zc2 = zc + np.array([delta, 0.0])
out.check("halving the loss", -np.log(softmax(zc2)[0]), -0.5 * np.log(softmax(zc)[0]))
out.val("pd_sqrt", pn, 3); out.val("pd_odds_new", pn / (1 - pn), 3); out.val("pd_odds_old", pp / (1 - pp), 2)
out.val("pd_ratio", (pn / (1 - pn)) / (pp / (1 - pp)), 2); out.val("pd_delta", delta, 2)
out.val("ln4", np.log(4), 3)
