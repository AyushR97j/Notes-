"""Chapter 10: generative and representation learning. Linear autoencoder vs PCA,
Gaussian KL vs torch.distributions, reparameterisation gradients, the optimal
GAN discriminator and JSD, DDPM forward process, InfoNCE / CLIP loss, skip-gram
with negative sampling (gradient vs autograd), ViT patch embedding counts."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm
from sklearn.decomposition import PCA
from common import setup, thousands

out = setup(__file__)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

# ------------------------------------------------- linear autoencoder = PCA
n, d, k = 500, 10, 3
Z = rng.normal(size=(n, 3)) @ rng.normal(size=(3, d)) + 0.1 * rng.normal(size=(n, d))
Xc = Z - Z.mean(0)
pca = PCA(k).fit(Xc)
err_pca = np.mean((pca.inverse_transform(pca.transform(Xc)) - Xc) ** 2)
torch.manual_seed(0)
ae = nn.Sequential(nn.Linear(d, k, bias=False), nn.Linear(k, d, bias=False))
opt = torch.optim.Adam(ae.parameters(), lr=0.01)
Xt = torch.tensor(Xc)
for _ in range(3000):
    opt.zero_grad(); loss = F.mse_loss(ae(Xt), Xt); loss.backward(); opt.step()
out.check("linear autoencoder reconstruction error -> PCA error", loss.item(), err_pca, rtol=1e-3, atol=1e-6)
out.val("ae_err", loss.item(), 5); out.val("pca_err", err_pca, 5)
# the decoder spans the PCA subspace: principal angles ~ 0
Wd = ae[1].weight.detach().numpy()                      # (d, k)
Qd, _ = np.linalg.qr(Wd)
cosang = np.linalg.svd(Qd.T @ pca.components_.T, compute_uv=False)
out.check("decoder column space = top-k principal subspace", cosang, np.ones(k), atol=1e-3)

# ------------------------------------------------------------ VAE pieces
# [[kl]]
def kl_gauss_std(mu, logvar):          # KL( N(mu, sigma^2) || N(0, 1) ), per dimension
    return 0.5 * (mu ** 2 + np.exp(logvar) - 1 - logvar)
# [[/kl]]
mu, sigma = 1.0, 0.5
kl = kl_gauss_std(mu, math.log(sigma ** 2))
ref = torch.distributions.kl_divergence(torch.distributions.Normal(mu, sigma), torch.distributions.Normal(0.0, 1.0)).item()
out.check("Gaussian KL closed form vs torch.distributions", kl, ref)
out.val("kl", kl, 4); out.val("kl0", kl_gauss_std(0.0, 0.0), 1)
mus = rng.normal(size=8); lvs = rng.normal(size=8) * 0.5
out.check("8-dim KL vs torch", kl_gauss_std(mus, lvs).sum(),
          torch.distributions.kl_divergence(torch.distributions.Normal(torch.tensor(mus), torch.tensor(np.exp(lvs / 2))),
                                            torch.distributions.Normal(0.0, 1.0)).sum().item())
# reparameterisation: d/dmu, d/dsigma of E[z^2], z = mu + sigma * eps
m_t = torch.tensor(mu, requires_grad=True); s_t = torch.tensor(sigma, requires_grad=True)
eps = torch.randn(200000, generator=torch.Generator().manual_seed(0))
zsmp = m_t + s_t * eps
(zsmp ** 2).mean().backward()
out.check("reparameterised MC gradient wrt mu ~ 2 mu", m_t.grad.item(), 2 * mu, atol=0.01)
out.check("reparameterised MC gradient wrt sigma ~ 2 sigma", s_t.grad.item(), 2 * sigma, atol=0.01)
out.val("rep_gmu", m_t.grad.item(), 3); out.val("rep_gsig", s_t.grad.item(), 3)

# ---------------------------------------------------------- GAN optimum
xs = np.linspace(-10, 10, 20001); dx = xs[1] - xs[0]
pd_ = norm.pdf(xs, 0, 1); pg = norm.pdf(xs, 1.5, 1)
Dstar = pd_ / (pd_ + pg)
V = np.sum(pd_ * np.log(Dstar) + pg * np.log(1 - Dstar)) * dx
jsd = jensenshannon(pd_, pg, base=math.e) ** 2           # scipy returns the JS distance
out.check("V(D*, G) = -log 4 + 2 JSD", V, -math.log(4) + 2 * jsd, atol=1e-6)
out.val("gan_V", V, 4); out.val("gan_jsd", jsd, 4); out.val("log4", math.log(4), 4)
out.val("dstar_0", Dstar[np.searchsorted(xs, 0.0)], 3); out.val("dstar_15", Dstar[np.searchsorted(xs, 1.5)], 3)
out.dat("gan", {"x": xs[::200], "pdata": pd_[::200], "pg": pg[::200], "dstar": Dstar[::200]})

# ------------------------------------------------------------- DDPM forward
T = 1000
betas = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1 - betas)
# [[ddpm]]
def q_sample(x0, t, eps):              # closed form of t noising steps at once
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps
# [[/ddpm]]
x0 = 2.0
xsim = np.full(100000, x0)
for t in range(300):
    xsim = np.sqrt(1 - betas[t]) * xsim + np.sqrt(betas[t]) * rng.normal(size=xsim.shape)
out.check("300 noising steps: mean matches sqrt(abar) x0", xsim.mean(), math.sqrt(abar[299]) * x0, atol=0.01)
out.check("300 noising steps: variance matches 1 - abar", xsim.var(), 1 - abar[299], atol=0.01)
cl = q_sample(x0, 299, rng.normal(size=100000))
out.val("abar_300", abar[299], 4); out.val("abar_T", abar[-1], 6); out.val("abar_1", abar[0], 4)
out.dat("ddpm", {"t": np.arange(1, T + 1)[::10], "signal": np.sqrt(abar)[::10], "noise": np.sqrt(1 - abar)[::10]})
out.val("snr_half", int(np.argmin(np.abs(abar - 0.5))) + 1)

# --------------------------------------------------------- InfoNCE / CLIP
B, D = 4, 8
img = F.normalize(torch.tensor(rng.normal(size=(B, D))), dim=1)
txt = F.normalize(img + 0.3 * torch.tensor(rng.normal(size=(B, D))), dim=1)
tau = 0.1
# [[clip]]
def clip_loss(img, txt, tau):
    logits = img @ txt.T / tau                     # (B, B): pair i with every caption j
    labels = torch.arange(len(img))                # the matching caption is on the diagonal
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
# [[/clip]]
lc = clip_loss(img, txt, tau).item()
S = (img @ txt.T / tau).numpy()
man = 0.5 * (np.mean([np.log(np.exp(S[i]).sum()) - S[i, i] for i in range(B)]) +
             np.mean([np.log(np.exp(S[:, j]).sum()) - S[j, j] for j in range(B)]))
out.check("CLIP loss by hand vs F.cross_entropy form", man, lc)
out.val("clip_loss", lc, 4); out.val("clip_lnB", math.log(B), 4)
out.val("clip_diag", ", ".join(f"{v:.2f}" for v in np.diag(S * tau)))

# -------------------------------------------------- skip-gram, negative sampling
Vv, Dw, kneg = 20, 5, 3
Win = rng.normal(size=(Vv, Dw)) * 0.3; Wout = rng.normal(size=(Vv, Dw)) * 0.3
c, o, negs = 4, 7, np.array([1, 12, 15])
sig = lambda z: 1 / (1 + np.exp(-z))
# [[sgns]]
def sgns_loss_grad(Win, Wout, c, o, negs):
    v = Win[c]                                         # centre word vector
    u_pos, u_neg = Wout[o], Wout[negs]                 # context and negative "output" vectors
    s_pos, s_neg = sig(u_pos @ v), sig(-u_neg @ v)
    loss = -np.log(s_pos) - np.log(s_neg).sum()
    g_v = -(1 - s_pos) * u_pos + ((1 - s_neg)[:, None] * u_neg).sum(0)
    g_pos = -(1 - s_pos) * v
    g_neg = (1 - s_neg)[:, None] * v[None, :]
    return loss, g_v, g_pos, g_neg
# [[/sgns]]
L_, g_v, g_pos, g_neg = sgns_loss_grad(Win, Wout, c, o, negs)
Wi = torch.tensor(Win, requires_grad=True); Wo = torch.tensor(Wout, requires_grad=True)
lt = -F.logsigmoid(Wo[o] @ Wi[c]) - F.logsigmoid(-(Wo[negs] @ Wi[c])).sum()
lt.backward()
out.check("SGNS loss vs torch", L_, lt.item())
out.check("SGNS grad wrt centre vector vs autograd", g_v, Wi.grad[c].numpy())
out.check("SGNS grad wrt context vector vs autograd", g_pos, Wo.grad[o].numpy())
out.check("SGNS grad wrt negatives vs autograd", g_neg, Wo.grad[negs].numpy())
out.val("sgns_loss", L_, 4)

# ------------------------------------------------------------------ ViT
P_, C_, H_, dm = 16, 3, 224, 768
patch = nn.Conv2d(C_, dm, P_, P_)
n_tok = (H_ // P_) ** 2
y_ = patch(torch.zeros(1, C_, H_, H_)).flatten(2).transpose(1, 2)
out.check("ViT patch tokens", list(y_.shape), [1, n_tok, dm], atol=0)
out.check("ViT patch embedding params", sum(p.numel() for p in patch.parameters()), P_ * P_ * C_ * dm + dm, atol=0)
out.val("vit_tokens", n_tok); out.val("vit_patch_params", thousands(P_ * P_ * C_ * dm + dm))
out.val("vit_patch_dim", P_ * P_ * C_)
out.val("vit_tokens_8", (H_ // 8) ** 2)

# ------------------------------------------------------------ problem values
out.val("pa1", kl_gauss_std(0.0, math.log(4.0)), 4); out.val("pa2", kl_gauss_std(2.0, 0.0), 1)
out.check("problem A KL vs torch", kl_gauss_std(0.0, math.log(4.0)),
          torch.distributions.kl_divergence(torch.distributions.Normal(0.0, 2.0), torch.distributions.Normal(0.0, 1.0)).item())
out.val("pd_patches", (224 // 14) ** 2); out.val("pd_tokens", (224 // 14) ** 2 + 1)
out.val("pe_noise", math.sqrt(0.75), 3); out.val("pe_snr", "1/3")
