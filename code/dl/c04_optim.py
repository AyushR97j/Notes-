"""Chapter 4: optimisers. GD on an ill-conditioned quadratic (rates, paths),
SGD/momentum/Nesterov/AdaGrad/RMSProp/Adam/AdamW from scratch vs torch.optim,
a hand Adam step, LR schedules vs torch schedulers, minibatch-gradient variance
vs batch size, gradient clipping, training curves, iterations per epoch."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons
from common import setup
from dlutil import sci

out = setup(__file__)
torch.set_default_dtype(torch.float64)

# --------------------------------------------- quadratic f = 0.5 (x^2 + k y^2)
kappa = 10.0
H = np.diag([1.0, kappa])
f = lambda p: 0.5 * p @ H @ p
grad = lambda p: H @ p
out.val("kappa", kappa, 0)
out.val("lr_max_stable", 2 / kappa, 2)
eta_opt = 2 / (1 + kappa)
rate_gd = (kappa - 1) / (kappa + 1)
rate_hb = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)
out.val("eta_opt", eta_opt, 4); out.val("rate_gd", rate_gd, 4); out.val("rate_hb", rate_hb, 4)
out.val("iters_gd", math.ceil(math.log(1e-6) / math.log(rate_gd)))
out.val("iters_hb", math.ceil(math.log(1e-6) / math.log(rate_hb)))
# empirical check of the GD rate with the optimal step
p = np.array([1.0, 1.0]); norms = []
for _ in range(50):
    p = p - eta_opt * grad(p); norms.append(np.linalg.norm(p))
emp = (norms[-1] / norms[-11]) ** (1 / 10)
out.check("GD contraction factor (kappa-1)/(kappa+1)", emp, rate_gd, atol=1e-9)
# heavy ball with optimal parameters
eta_hb = 4 / (1 + math.sqrt(kappa)) ** 2; beta_hb = rate_hb ** 2
p = np.array([1.0, 1.0]); pp = p.copy(); norms = []
for _ in range(200):
    p, pp = p - eta_hb * grad(p) + beta_hb * (p - pp), p
    norms.append(np.linalg.norm(p))
emp = (norms[-1] / norms[-51]) ** (1 / 50)
# at the optimal parameters the iteration matrix has a double eigenvalue, so the
# error decays like t * rate^t; correct the 50-step ratio for the factor t
out.check("heavy-ball rate (sqrt k - 1)/(sqrt k + 1)", emp, rate_hb * (200 / 150) ** (1 / 50), atol=5e-4)
# divergence above 2/lambda_max
p = np.array([1.0, 1.0])
for _ in range(20):
    p = p - 0.21 * grad(p)
out.val("diverge_y", abs(p[1]), 1)

# ------------------------------------------------- optimisers from scratch
# [[optimisers]]
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
# [[/optimisers]]

def run_ours(opt, p0, steps, gfun):
    p = p0.copy(); traj = [p.copy()]
    for _ in range(steps):
        p = opt.step(p, gfun(p)); traj.append(p.copy())
    return np.array(traj)

def run_torch(make, p0, steps, gfun):
    t = torch.tensor(p0, requires_grad=True); opt = make([t]); traj = [p0.copy()]
    for _ in range(steps):
        opt.zero_grad(); t.grad = torch.tensor(gfun(t.detach().numpy())); opt.step()
        traj.append(t.detach().numpy().copy())
    return np.array(traj)

rngq = np.random.default_rng(3)
A = rngq.normal(size=(5, 5)); Hq = A @ A.T / 5 + np.eye(5) * 0.1
gq = lambda p: Hq @ p + 0.3 * np.sin(p)            # a smooth non-quadratic gradient
p0 = rngq.normal(size=5)
pairs = {
    "SGD": (SGD(0.1), lambda ps: torch.optim.SGD(ps, lr=0.1)),
    "SGD momentum": (SGD(0.1, 0.9), lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9)),
    "Nesterov": (SGD(0.1, 0.9, True), lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9, nesterov=True)),
    "SGD + weight decay": (SGD(0.1, 0.9, weight_decay=0.01),
                           lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9, weight_decay=0.01)),
    "AdaGrad": (AdaGrad(0.5), lambda ps: torch.optim.Adagrad(ps, lr=0.5)),
    "RMSProp": (RMSProp(0.01), lambda ps: torch.optim.RMSprop(ps, lr=0.01)),
    "Adam": (Adam(0.05), lambda ps: torch.optim.Adam(ps, lr=0.05)),
    "Adam + L2": (Adam(0.05, weight_decay=0.1), lambda ps: torch.optim.Adam(ps, lr=0.05, weight_decay=0.1)),
    "AdamW": (Adam(0.05, weight_decay=0.1, decoupled=True),
              lambda ps: torch.optim.AdamW(ps, lr=0.05, weight_decay=0.1)),
}
for name, (ours, make) in pairs.items():
    out.check(f"{name}: 25 steps vs torch.optim", run_ours(ours, p0, 25, gq), run_torch(make, p0, 25, gq))

# ------------------------------------------------------- hand Adam step
g1 = np.array([0.5, -0.02]); g2 = np.array([0.3, 0.04])
lr = 0.001
adam = Adam(lr)
th0 = np.array([1.0, 1.0])
th1 = adam.step(th0, g1)
m1, v1 = adam.m.copy(), adam.v.copy()
th2 = adam.step(th1, g2)
m2, v2 = adam.m.copy(), adam.v.copy()
t = torch.tensor(th0, requires_grad=True); o = torch.optim.Adam([t], lr=lr)
for g in (g1, g2):
    o.zero_grad(); t.grad = torch.tensor(g); o.step()
out.check("hand Adam two steps vs torch", th2, t.detach().numpy())
for i in range(2):
    out.val(f"a_m1_{i}", m1[i], 4); out.val(f"a_v1_{i}", sci(v1[i], 2))
    out.val(f"a_m2_{i}", m2[i], 4); out.val(f"a_v2_{i}", sci(v2[i], 3))
    out.val(f"a_mhat2_{i}", m2[i] / (1 - 0.9 ** 2), 4)
    out.val(f"a_vhat2_{i}", sci(v2[i] / (1 - 0.999 ** 2), 3))
    out.val(f"a_step1_{i}", sci(th0[i] - th1[i], 3)); out.val(f"a_step2_{i}", sci(th1[i] - th2[i], 3))
    out.val(f"a_th2_{i}", th2[i], 6)
out.val("a_bc1_2", 1 - 0.9 ** 2, 2); out.val("a_bc2_2", 1 - 0.999 ** 2, 6)
# without bias correction the first step is tiny
m_nb = 0.1 * g1; v_nb = 0.001 * g1 ** 2
out.val("a_nobc_step", sci(lr * m_nb[0] / (np.sqrt(v_nb[0]) + 1e-8), 2))
# SGD momentum hand steps (PyTorch convention)
sg = SGD(0.1, 0.9); q0 = np.array([1.0])
q1 = sg.step(q0, np.array([2.0])); b1 = sg.buf.copy()
q2 = sg.step(q1, np.array([1.0])); b2 = sg.buf.copy()
out.val("mom_b1", b1[0], 2); out.val("mom_q1", q1[0], 2); out.val("mom_b2", b2[0], 2); out.val("mom_q2", q2[0], 2)

# -------------------------------------------------- optimiser paths on f
start = np.array([-4.0, 1.5])
paths = {
    "gd": run_ours(SGD(0.18), start, 30, grad),
    "mom": run_ours(SGD(0.03, 0.9), start, 30, grad),
    "adam": run_ours(Adam(0.4), start, 30, grad),
}
for k, tr in paths.items():
    out.dat(f"path_{k}", {"x": tr[:, 0], "y": tr[:, 1]})
    out.val(f"path_{k}_f", f(tr[-1]), 4)
xs, ys = [], []
for c in (0.5, 2, 4.5, 8, 12.5):
    tt = np.linspace(0, 2 * np.pi, 120)
    xs += list(np.sqrt(2 * c) * np.cos(tt)) + [np.nan]
    ys += list(np.sqrt(2 * c / kappa) * np.sin(tt)) + [np.nan]
out.dat("contours", {"x": np.array(xs), "y": np.array(ys)})

# -------------------------------------------------------- LR schedules
T, W, lr0 = 100, 10, 1.0
# [[schedules]]
def step_decay(t, lr0, step=30, gamma=0.1):
    return lr0 * gamma ** (t // step)

def cosine(t, lr0, T, lr_min=0.0):
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * t / T))

def warmup_cosine(t, lr0, W, T):
    if t < W:
        return lr0 * (t + 1) / W                     # linear warm-up
    return cosine(t - W, lr0, T - W)

def one_cycle(t, lr_max, T, pct=0.3, div=25.0, final_div=1e4):
    up = pct * T - 1
    lo, end = lr_max / div, lr_max / div / final_div
    cos_anneal = lambda a, b, frac: b + (a - b) / 2 * (1 + math.cos(math.pi * frac))
    if t <= up:
        return cos_anneal(lo, lr_max, t / up)
    return cos_anneal(lr_max, end, (t - up) / (T - 1 - up))
# [[/schedules]]
def torch_sched(make, steps):
    pp = torch.zeros(1, requires_grad=True)
    o = torch.optim.SGD([pp], lr=lr0)
    s = make(o); vals = []
    for _ in range(steps):
        vals.append(o.param_groups[0]["lr"]); o.step(); s.step()
    return np.array(vals)
ts = np.arange(T)
sd = np.array([step_decay(t, lr0) for t in ts])
co = np.array([cosine(t, lr0, T) for t in ts])
oc = np.array([one_cycle(t, lr0, T) for t in ts])
wc = np.array([warmup_cosine(t, lr0, W, T) for t in ts])
out.check("step decay vs StepLR", sd, torch_sched(lambda o: torch.optim.lr_scheduler.StepLR(o, 30, 0.1), T))
out.check("cosine vs CosineAnnealingLR", co, torch_sched(lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T), T))
out.check("one-cycle vs OneCycleLR", oc,
          torch_sched(lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=lr0, total_steps=T,
                                                                     cycle_momentum=False), T))
out.dat("sched", {"t": ts, "step": sd, "cos": co, "wcos": wc, "onecycle": oc})
out.val("cos_t50", cosine(50, 0.1, 100), 4); out.val("cos_t25", cosine(25, 0.1, 100), 4)
out.val("cos_t75", cosine(75, 0.1, 100), 4)
out.val("step_t65", step_decay(65, 0.1), 4)

# ------------------------------------------- gradient noise vs batch size
rng = np.random.default_rng(0)
N, d = 20000, 20
Xn = rng.normal(size=(N, d)); wtrue = rng.normal(size=d)
yn = (rng.random(N) < 1 / (1 + np.exp(-Xn @ wtrue))).astype(float)
w0 = np.zeros(d)
def g_batch(idx):
    pz = 1 / (1 + np.exp(-Xn[idx] @ w0))
    return Xn[idx].T @ (pz - yn[idx]) / len(idx)
gfull = g_batch(np.arange(N))
Bs = [1, 4, 16, 64, 256, 1024]
var = []
for B in Bs:
    reps = 3000 if B < 256 else 600
    e = [np.sum((g_batch(rng.integers(0, N, B)) - gfull) ** 2) for _ in range(reps)]
    var.append(np.mean(e))
var = np.array(var)
slope = np.polyfit(np.log(Bs), np.log(var), 1)[0]
out.dat("noise", {"B": Bs, "var": var}, nd=10)
out.val("noise_slope", slope, 3)
out.check("minibatch-gradient variance scales like 1/B (slope)", slope, -1.0, atol=0.05)
out.val("noise_ratio", var[0] / var[3], 1)

# --------------------------------------------------- clipping
g = torch.tensor([3.0, 4.0]); pz = torch.zeros(2, requires_grad=True); pz.grad = g.clone()
torch.nn.utils.clip_grad_norm_([pz], 1.0)
ours = np.array([3.0, 4.0]) * min(1.0, 1.0 / 5.0)
out.check("clip by norm vs torch", ours, pz.grad.numpy())
pz.grad = g.clone(); torch.nn.utils.clip_grad_value_([pz], 1.0)
out.check("clip by value vs torch", np.clip([3.0, 4.0], -1, 1), pz.grad.numpy())

# -------------------------------------------------- epochs and iterations
out.val("ep_N", 50000); out.val("ep_B", 128)
out.val("ep_iters", math.ceil(50000 / 128)); out.val("ep_total", 30 * math.ceil(50000 / 128))

# ---------------------------------------------- training curves on two moons
Xm, ym = make_moons(1024, noise=0.25, random_state=0)
Xm = torch.tensor(Xm); ym = torch.tensor(ym)
def train_curve(make, steps=400, B=64, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 2))
    opt = make(net.parameters()); g = torch.Generator().manual_seed(seed); losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(Xm), (B,), generator=g)
        opt.zero_grad(); loss = F.cross_entropy(net(Xm[idx]), ym[idx]); loss.backward(); opt.step()
        with torch.no_grad():
            losses.append(F.cross_entropy(net(Xm), ym).item())
    return np.array(losses)
curves = {
    "sgd": train_curve(lambda ps: torch.optim.SGD(ps, lr=0.05)),
    "mom": train_curve(lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9)),
    "rms": train_curve(lambda ps: torch.optim.RMSprop(ps, lr=0.003)),
    "adam": train_curve(lambda ps: torch.optim.Adam(ps, lr=0.003)),
}
cols = {"step": np.arange(1, 401)[::4]}
for k, v in curves.items():
    cols[k] = v[::4]
    out.val(f"curve_{k}_final", v[-1], 3)
    out.val(f"curve_{k}_100", v[99], 3)
out.dat("curves", cols)

# ------------------------------------------------ values used in the problems
out.val("p_steps_ep", math.ceil(60000 / 256)); out.val("p_last", 60000 - 256 * (60000 // 256))
out.val("p_full", 60000 // 256); out.val("p_steps10", 10 * math.ceil(60000 / 256))
out.val("p_bc10", 1 - 0.999 ** 10, 5); out.val("p_bc10_inv", 1 / (1 - 0.999 ** 10), 0)
out.val("p_bc10_sqrt", math.sqrt(1 / (1 - 0.999 ** 10)), 0)
out.val("p_t09", math.log(0.1) / math.log(0.9), 1); out.val("p_t099", math.log(0.1) / math.log(0.99), 1)
out.val("p_n09", math.ceil(math.log(0.1) / math.log(0.9))); out.val("p_n099", math.ceil(math.log(0.1) / math.log(0.99)))
out.val("nobc_factor", 0.1 / math.sqrt(0.001), 2)
gcl = np.array([6.0, -8.0, 0.0])
t3 = torch.zeros(3, requires_grad=True); t3.grad = torch.tensor(gcl)
torch.nn.utils.clip_grad_norm_([t3], 5.0)
out.check("problem: clip (6,-8,0) by norm 5", gcl * 5 / 10, t3.grad.numpy())
fmtv = lambda v: "(" + ",\\,".join(f"{x:g}" for x in v) + ")"
out.val("clipn_p", fmtv(gcl * 5 / 10)); out.val("clipv_p", fmtv(np.clip(gcl, -5, 5)))
out.val("clipn_ex", fmtv(np.array([3.0, 4.0]) / 5)); out.val("clipv_ex", fmtv(np.clip([3.0, 4.0], -1, 1)))
