"""Chapter 5b: regularisation. Inverted dropout (vs torch semantics), the effect
of lambda in E + lambda * sum w^2 (polynomial ridge vs sklearn), Adam+L2 vs AdamW,
over/underfitting curves and early stopping, generalisation gap."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from common import setup

out = setup(__file__)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

# --------------------------------------------------------------- dropout
# [[dropout]]
def dropout(h, p, train, rng):
    if not train or p == 0:
        return h                                    # inference: identity
    mask = (rng.random(h.shape) >= p)               # keep with probability 1-p
    return h * mask / (1 - p)                       # inverted scaling
# [[/dropout]]
h = np.array([2.0, 4.0, 6.0, 8.0])
mask_demo = np.array([1, 0, 1, 0])
out.val("drop_demo", "(" + ",\\,".join(f"{v:g}" for v in h * mask_demo / 0.5) + ")")
samples = np.array([dropout(h, 0.5, True, rng) for _ in range(200000)])
out.check("inverted dropout preserves the mean", samples.mean(0), h, atol=0.03)
out.check("dropout at inference is identity", dropout(h, 0.5, False, rng), h)
# torch: same expectation, identity in eval, kept entries scaled by 1/(1-p)
drop = torch.nn.Dropout(0.5)
yt = drop(torch.tensor(h).repeat(1000, 1))
kept = yt[yt != 0]
out.check("torch dropout scales kept units by 1/(1-p)", np.unique(np.round((kept / torch.tensor(h).repeat(1000, 1)[yt != 0]).numpy(), 12)), [2.0])
drop.eval()
out.check("torch dropout eval = identity", drop(torch.tensor(h)).numpy(), h)
out.val("drop_var", np.var(samples[:, 0]), 2)          # Var = h^2 p/(1-p) = 4

# ------------------------------------------------ lambda in E + lambda sum w^2
xtr = np.sort(rng.uniform(-1, 1, 12)); ytr = np.sin(np.pi * xtr) + rng.normal(0, 0.25, 12)
xte = np.linspace(-1, 1, 400); yte = np.sin(np.pi * xte)
deg = 9
Phi = lambda x: np.vander(x, deg + 1, increasing=True)[:, 1:]   # no constant column
# [[ridge]]
def fit_l2(Phi_x, y, lam):
    """Minimise sum (y - b - Phi w)^2 + lam * ||w||^2 (bias not penalised)."""
    mu_x, mu_y = Phi_x.mean(0), y.mean()
    Xc = Phi_x - mu_x
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ (y - mu_y))
    return w, mu_y - mu_x @ w
# [[/ridge]]
lams = 10.0 ** np.arange(-8, 2.01, 0.5)
tr, te, nrm = [], [], []
for lam in lams:
    w, b = fit_l2(Phi(xtr), ytr, lam)
    ref = Ridge(alpha=lam, solver="cholesky").fit(Phi(xtr), ytr)
    out.check(f"ridge lambda={lam:.0e} vs sklearn", np.r_[w, b], np.r_[ref.coef_, ref.intercept_], atol=1e-6, rtol=1e-4)
    tr.append(np.mean((Phi(xtr) @ w + b - ytr) ** 2)); te.append(np.mean((Phi(xte) @ w + b - yte) ** 2))
    nrm.append(np.linalg.norm(w))
tr, te, nrm = map(np.array, (tr, te, nrm))
out.dat("ridge", {"lam": lams, "train": tr, "test": te, "norm": nrm}, nd=10)
ib = int(np.argmin(te))
out.val("ridge_best_lam", f"10^{{{np.log10(lams[ib]):.1f}}}".replace(".0}", "}"))
out.val("ridge_best_te", te[ib], 3)
for key, lam in (("small", 1e-8), ("best", lams[ib]), ("big", 10.0)):
    w, b = fit_l2(Phi(xtr), ytr, lam)
    out.val(f"ridge_{key}_tr", np.mean((Phi(xtr) @ w + b - ytr) ** 2), 3)
    out.val(f"ridge_{key}_te", np.mean((Phi(xte) @ w + b - yte) ** 2), 3)
    out.val(f"ridge_{key}_norm", np.linalg.norm(w), 1)
    out.dat(f"fit_{key}", {"x": xte, "y": Phi(xte) @ w + b})
out.dat("pts", {"x": xtr, "y": ytr})
out.dat("truth", {"x": xte, "y": yte})

# gradient-descent view: one step of SGD with L2 shrinks w by (1 - eta*lam)
eta, lam = 0.1, 0.01
out.val("wd_factor", 1 - eta * 2 * lam, 3)        # d/dw lam w^2 = 2 lam w

# --------------------------------------------------------- Adam+L2 vs AdamW
def decay_run(opt_name, wd, steps=2000, lr=1e-3):
    torch.manual_seed(0)
    p = torch.ones(2, requires_grad=True)
    opt = (torch.optim.Adam if opt_name == "adam" else torch.optim.AdamW)([p], lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        opt.zero_grad()
        noise = torch.randn(1, generator=g).item()
        p.grad = torch.tensor([noise * 1.0, 0.0])    # param 0: noisy loss gradient; param 1: none
        opt.step()
    return p.detach().numpy()
for name in ("adam", "adamw"):
    for wd in (0.01, 0.1):
        r = decay_run(name, wd)
        out.val(f"{name}_{str(wd).replace('.', '')}_p0", r[0], 3)
        out.val(f"{name}_{str(wd).replace('.', '')}_p1", r[1], 3)
out.val("adamw_pred_001", (1 - 1e-3 * 0.01) ** 2000, 3)
out.val("adamw_pred_01", (1 - 1e-3 * 0.1) ** 2000, 3)

# ------------------------------------------- over/underfitting from curves
n_tr = 24
Xo = torch.tensor(rng.uniform(-1, 1, (n_tr + 400, 1)))
yo = torch.sin(3 * Xo) + 0.3 * torch.tensor(rng.normal(size=(n_tr + 400, 1)))
Xtr_, ytr_, Xva, yva = Xo[:n_tr], yo[:n_tr], Xo[n_tr:], yo[n_tr:]
def curves(width, depth, epochs, lr=3e-3, wd=0.0, p=0.0):
    torch.manual_seed(0)
    layers, d = [], 1
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.ReLU()] + ([torch.nn.Dropout(p)] if p else [])
        d = width
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    trl, val = [], []
    for _ in range(epochs):
        net.train(); opt.zero_grad(); F.mse_loss(net(Xtr_), ytr_).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            trl.append(F.mse_loss(net(Xtr_), ytr_).item()); val.append(F.mse_loss(net(Xva), yva).item())
    return np.array(trl), np.array(val)
E = 800
tr_o, va_o = curves(128, 3, E)
tr_u, va_u = curves(2, 1, E, lr=1e-2)
tr_r, va_r = curves(128, 3, E, p=0.2)
ep = np.arange(1, E + 1)[::5]
out.dat("over", {"ep": ep, "tr": tr_o[::5], "va": va_o[::5], "tru": tr_u[::5], "vau": va_u[::5],
                 "trr": tr_r[::5], "var": va_r[::5]})
best = int(np.argmin(va_o))
out.val("es_epoch", best + 1); out.val("es_val", va_o[best], 3); out.val("final_val", va_o[-1], 3)
out.val("final_tr", tr_o[-1], 3); out.val("gap_final", va_o[-1] - tr_o[-1], 3)
out.val("noise_var", 0.09, 2)
out.val("u_tr", tr_u[-1], 3); out.val("u_va", va_u[-1], 3)
out.val("r_tr", tr_r[-1], 3); out.val("r_va", va_r[-1], 3)
# early stopping with patience 100
pat, best_v, best_e, stop = 100, np.inf, 0, E
for e, v in enumerate(va_o):
    if v < best_v - 1e-6:
        best_v, best_e = v, e
    elif e - best_e >= pat:
        stop = e + 1
        break
out.val("es_stop", stop); out.val("es_pat", pat)
