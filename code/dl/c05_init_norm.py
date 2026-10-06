"""Chapter 5a: initialisation (variance through depth, Xavier/He vs torch.nn.init)
and normalisation (BatchNorm forward/backward/inference, LayerNorm, GroupNorm,
RMSNorm), all checked against PyTorch; BN vs no-BN training curves."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from common import setup

out = setup(__file__)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

# ----------------------------------------------------------- init formulas
fan_in, fan_out = 256, 128
xav_u = math.sqrt(6 / (fan_in + fan_out)); xav_n = math.sqrt(2 / (fan_in + fan_out))
he_n = math.sqrt(2 / fan_in); he_u = math.sqrt(6 / fan_in)
out.val("xav_u", xav_u, 4); out.val("xav_n", xav_n, 4); out.val("he_n", he_n, 4); out.val("he_u", he_u, 4)
W = torch.empty(fan_out, fan_in)
torch.nn.init.xavier_uniform_(W)
out.check("xavier_uniform bound (max |w| <= bound, ~ bound)", W.abs().max().item(), xav_u, atol=2e-3)
torch.nn.init.kaiming_normal_(W, nonlinearity="relu")
out.check("kaiming_normal std", W.std().item(), he_n, atol=2e-3)
torch.nn.init.xavier_normal_(W)
out.check("xavier_normal std", W.std().item(), xav_n, atol=2e-3)
# PyTorch's default nn.Linear init: kaiming_uniform with a=sqrt(5) -> U(-1/sqrt(fan_in), 1/sqrt(fan_in))
lin = torch.nn.Linear(fan_in, fan_out)
out.check("nn.Linear default bound 1/sqrt(fan_in)", lin.weight.abs().max().item(), 1 / math.sqrt(fan_in), atol=2e-3)
out.val("default_bound", 1 / math.sqrt(fan_in), 4)
out.val("default_std", 1 / math.sqrt(3 * fan_in), 4)

# ------------------------------------- activation std through 50 layers
def act_std_profile(act, std_fn, depth=50, width=256, B=128, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(B, width)
    stds = []
    for _ in range(depth):
        Wl = torch.randn(width, width) * std_fn(width)
        h = act(h @ Wl.T)
        stds.append(h.std().item())
    return np.array(stds)
prof = {
    "small_tanh": act_std_profile(torch.tanh, lambda n: 0.01),
    "xav_tanh": act_std_profile(torch.tanh, lambda n: math.sqrt(1 / n)),
    "xav_relu": act_std_profile(torch.relu, lambda n: math.sqrt(1 / n)),
    "he_relu": act_std_profile(torch.relu, lambda n: math.sqrt(2 / n)),
    "big_relu": act_std_profile(torch.relu, lambda n: math.sqrt(3 / n)),
}
cols = {"layer": np.arange(1, 51)}
for k, v in prof.items():
    cols[k] = np.log10(v)
    out.val(f"std50_{k}", v[-1], 3 if v[-1] > 1e-3 else 8)
out.dat("actstd", cols)
out.val("std50_xav_relu_pred", (1 / math.sqrt(2)) ** 49 * prof["xav_relu"][0], 8)
out.val("std10_small_tanh", prof["small_tanh"][9], 10)
from dlutil import sci
out.val("std10_small_tanh_sci", sci(prof["small_tanh"][9], 1))
out.val("std50_xav_relu_sci", sci(prof["xav_relu"][-1], 1))
out.val("std50_big_relu_sci", sci(prof["big_relu"][-1], 1))
# second moment of ReLU output: E[relu(z)^2] = Var(z)/2 for symmetric z
zz = rng.normal(0, 2.0, 2_000_000)
out.check("E[relu(z)^2] = Var(z)/2", np.mean(np.maximum(zz, 0) ** 2), 4.0 / 2, atol=1e-2)

# ------------------------------------------------------------- BatchNorm
# [[bn]]
def bn_forward(x, gamma, beta, eps=1e-5):          # x: (B, C), training mode
    mu = x.mean(0)
    var = x.var(0)                                 # biased (divide by B)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, var, eps)

def bn_backward(dy, gamma, cache):
    xhat, var, eps = cache
    B = dy.shape[0]
    dgamma = (dy * xhat).sum(0)
    dbeta = dy.sum(0)
    dxhat = dy * gamma
    dx = (B * dxhat - dxhat.sum(0) - xhat * (dxhat * xhat).sum(0)) / (B * np.sqrt(var + eps))
    return dx, dgamma, dbeta
# [[/bn]]
xb = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [6.0, 60.0]])
gamma = np.array([1.0, 2.0]); beta = np.array([0.0, 1.0])
yb, cache = bn_forward(xb, gamma, beta)
bn = torch.nn.BatchNorm1d(2)
with torch.no_grad():
    bn.weight.copy_(torch.tensor(gamma)); bn.bias.copy_(torch.tensor(beta))
xt = torch.tensor(xb, requires_grad=True)
yt = bn(xt)
out.check("BN forward (train) vs torch", yb, yt.detach().numpy())
dy = rng.normal(size=xb.shape)
dx, dg, dbt = bn_backward(dy, gamma, cache)
yt.backward(torch.tensor(dy))
out.check("BN backward dx vs autograd", dx, xt.grad.numpy())
out.check("BN backward dgamma vs autograd", dg, bn.weight.grad.numpy())
out.check("BN backward dbeta vs autograd", dbt, bn.bias.grad.numpy())
mu0 = xb[:, 0].mean(); var0 = xb[:, 0].var(); uvar0 = xb[:, 0].var(ddof=1)
out.val("bn_mu", mu0, 2); out.val("bn_var", var0, 3); out.val("bn_uvar", uvar0, 3)
out.val("bn_sd", math.sqrt(var0 + 1e-5), 4)
for i in range(4):
    out.val(f"bn_y{i}", yb[i, 0], 4)
    out.val(f"bn_y2{i}", yb[i, 1], 4)
# running statistics after one step (momentum 0.1, unbiased variance)
rm = 0.9 * 0 + 0.1 * xb.mean(0); rv = 0.9 * 1 + 0.1 * xb.var(0, ddof=1)
out.check("BN running_mean after 1 step", rm, bn.running_mean.numpy())
out.check("BN running_var after 1 step (unbiased)", rv, bn.running_var.numpy())
out.val("bn_rm0", rm[0], 2); out.val("bn_rv0", rv[0], 4)
bn.eval()
x_test = np.array([[3.0, 30.0]])
y_eval = gamma * (x_test - rm) / np.sqrt(rv + 1e-5) + beta
out.check("BN eval uses running stats", y_eval, bn(torch.tensor(x_test)).detach().numpy())
out.val("bn_eval0", y_eval[0, 0], 4)
y_trainmode_single = 0.0   # a batch of one is normalised to exactly beta
try:
    bn.train(); bn(torch.tensor(x_test))
    out.val("bn_b1_error", "no error")
except ValueError as e:
    out.val("bn_b1_error", "ValueError")
out.val("bn_params", sum(p.numel() for p in torch.nn.BatchNorm2d(64).parameters()))
out.val("bn_buffers", sum(b.numel() for n, b in torch.nn.BatchNorm2d(64).named_buffers() if "running" in n))

# --------------------------------------------- LayerNorm / RMSNorm / GroupNorm
x4 = np.array([1.0, 2.0, 3.0, 6.0])
# [[ln]]
def layer_norm(x, gamma, beta, eps=1e-5):         # normalise over the last axis
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def rms_norm(x, gamma, eps=1e-6):                 # no centring, no beta
    return gamma * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)

def group_norm(x, G, gamma, beta, eps=1e-5):      # x: (B, C, H, W)
    B, C, H, W = x.shape
    xg = x.reshape(B, G, C // G, H, W)
    mu = xg.mean((2, 3, 4), keepdims=True); var = xg.var((2, 3, 4), keepdims=True)
    xg = (xg - mu) / np.sqrt(var + eps)
    return xg.reshape(B, C, H, W) * gamma[None, :, None, None] + beta[None, :, None, None]
# [[/ln]]
ln = layer_norm(x4, np.ones(4), np.zeros(4))
out.check("LayerNorm vs torch", ln, F.layer_norm(torch.tensor(x4), (4,)).numpy())
rms = rms_norm(x4, np.ones(4))
out.check("RMSNorm vs torch.nn.RMSNorm", rms, torch.nn.RMSNorm(4, eps=1e-6)(torch.tensor(x4)).detach().numpy())
for i in range(4):
    out.val(f"ln{i}", ln[i], 4); out.val(f"rms{i}", rms[i], 4)
out.val("rms_val", math.sqrt((x4 ** 2).mean()), 4)
xg = rng.normal(size=(2, 6, 3, 3)); gg = rng.normal(size=6); bg = rng.normal(size=6)
out.check("GroupNorm vs torch", group_norm(xg, 3, gg, bg),
          F.group_norm(torch.tensor(xg), 3, torch.tensor(gg), torch.tensor(bg)).numpy())
out.check("GroupNorm(G=C) = InstanceNorm", group_norm(xg, 6, np.ones(6), np.zeros(6)),
          F.instance_norm(torch.tensor(xg)).numpy())
out.check("GroupNorm(G=1) = LayerNorm over (C,H,W)", group_norm(xg, 1, np.ones(6), np.zeros(6)),
          F.layer_norm(torch.tensor(xg), (6, 3, 3)).numpy())

# ---------------------------------------------- deep MLP with and without BN
from sklearn.datasets import make_classification
Xc, yc = make_classification(2000, 20, n_informative=10, random_state=0)
Xc = torch.tensor(Xc); yc = torch.tensor(yc)
def deep_net(bn, depth=10, width=64):
    layers, d = [], 20
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width)] + ([torch.nn.BatchNorm1d(width)] if bn else []) + [torch.nn.Sigmoid()]
        d = width
    layers += [torch.nn.Linear(d, 2)]
    return torch.nn.Sequential(*layers)
def train(bn, lr, steps=300):
    torch.manual_seed(0)
    net = deep_net(bn); opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    g = torch.Generator().manual_seed(0); losses = []
    for _ in range(steps):
        idx = torch.randint(0, 2000, (64,), generator=g)
        net.train(); opt.zero_grad(); F.cross_entropy(net(Xc[idx]), yc[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            losses.append(F.cross_entropy(net(Xc), yc).item())
    return np.array(losses)
cur = {"nobn": train(False, 0.05), "bn": train(True, 0.05), "bnhi": train(True, 0.5)}
out.dat("bncurve", {"step": np.arange(1, 301)[::3], **{k: v[::3] for k, v in cur.items()}})
for k, v in cur.items():
    out.val(f"bnc_{k}", v[-1], 3)

# ---------------------------------------------------- problem values
out.val("p_a", math.sqrt(6 / 1024), 4)
from common import thousands
m = torch.nn.Sequential(torch.nn.Linear(256, 128, bias=False), torch.nn.BatchNorm1d(128))
n = sum(p.numel() for p in m.parameters())
out.check("Linear(256,128,bias=False)+BN params", n, 256 * 128 + 2 * 128, atol=0)
out.val("p_e", thousands(n)); out.val("p_e_w", thousands(256 * 128))
