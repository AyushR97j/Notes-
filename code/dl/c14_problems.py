"""Chapter 14: every number in the problem set, each checked against PyTorch,
SciPy or a direct computation."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import setup, thousands
from dlutil import human, sci

out = setup(__file__)
torch.set_default_dtype(torch.float64)
T = torch.tensor
fv = lambda v, nd=3: ",\\allowbreak ".join(f"{x:.{nd}f}" for x in np.ravel(v))
npar = lambda m: sum(p.numel() for p in m.parameters())
sig = lambda z: 1 / (1 + np.exp(-z))
def softmax(z):
    e = np.exp(z - np.max(z)); return e / e.sum()

# ---------------------------------------------------------------- neurons
x = np.array([2.0, -1.0, 0.5]); w = np.array([0.3, 0.8, -1.2]); b = 0.5
z = w @ x + b
out.val("p1_z", z, 2); out.val("p1_relu", max(z, 0), 2); out.val("p1_sig", sig(z), 4)
out.check("P1 sigmoid", sig(z), torch.sigmoid(T(z)).item())
m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
n2 = 65 * 128 + 129 * 64 + 65 * 10
out.check("P2 MLP params", n2, npar(m), atol=0)
out.val("p2_a", thousands(65 * 128)); out.val("p2_b", thousands(129 * 64)); out.val("p2_c", 65 * 10); out.val("p2", thousands(n2))
wv, bv, xv, yv = np.array([1.0, -1.0]), 0.0, np.array([2.0, 1.0]), -1
marg = yv * (wv @ xv + bv)
out.val("p4_marg", marg, 0)
wn = wv + yv * xv; bn = bv + yv
out.val("p4_w", f"({wn[0]:g},{wn[1]:g})"); out.val("p4_b", bn, 0); out.val("p4_marg2", yv * (wn @ xv + bn), 0)

# --------------------------------------------------------------- softmax / CE
z6 = np.array([1.0, 2.0, 3.0]); p6 = softmax(z6)
out.check("P6 CE", -np.log(p6[2]), F.cross_entropy(T(z6)[None], T([2])).item())
out.val("p6_p", fv(p6)); out.val("p6_ce", -math.log(p6[2]), 4)
z7 = np.array([0.5, 1.5, -0.5, 2.0, 0.0]); p7 = softmax(z7)
out.check("P7 CE", -np.log(p7[1]), F.cross_entropy(T(z7)[None], T([1])).item())
out.val("p7_e", fv(np.exp(z7))); out.val("p7_sum", np.exp(z7).sum(), 3); out.val("p7_p1", p7[1], 4)
out.val("p7_ce", -math.log(p7[1]), 4); out.val("p7_pred", int(np.argmax(p7)) + 1); out.val("p7_pmax", p7.max(), 4)
out.val("p8_y0", -math.log(0.2), 4); out.val("p8_y1", -math.log(0.8), 4)
out.check("P8 BCE", -math.log(0.2), F.binary_cross_entropy(T([0.8]), T([0.0])).item())
out.val("p9_0", sig(0) * (1 - sig(0)), 4); out.val("p9_2", sig(2) * (1 - sig(2)), 4)
for Tt in (1.0, 0.5, 2.0):
    out.val(f"p11_{str(Tt).replace('.', '')}", softmax(np.array([2.0, 0.0]) / Tt)[0], 4)
q12 = np.full(4, 0.2 / 4); q12[0] += 0.8
p12 = np.array([0.7, 0.1, 0.1, 0.1])
out.val("p12_q", fv(q12, 2)); out.val("p12_ce", -np.sum(q12 * np.log(p12)), 4); out.val("p12_ce_hard", -math.log(0.7), 4)
out.check("P12 smoothed CE vs torch", -np.sum(q12 * np.log(p12)),
          F.cross_entropy(T(np.log(p12))[None], T([0]), label_smoothing=0.2).item())

# --------------------------------------------------------------- backprop
x13, w1, b1, w2, b2, y13, eta = 2.0, 0.5, 0.0, -1.0, 0.5, 1.0, 0.1
z1 = w1 * x13 + b1; h1 = max(z1, 0); yh = w2 * h1 + b2; L = 0.5 * (yh - y13) ** 2
d2 = yh - y13; gw2 = d2 * h1; gb2 = d2; d1 = d2 * w2 * (z1 > 0); gw1 = d1 * x13; gb1 = d1
P = [T(v, requires_grad=True) for v in (w1, b1, w2, b2)]
yt = P[2] * torch.relu(P[0] * x13 + P[1]) + P[3]
(0.5 * (yt - y13) ** 2).backward()
out.check("P13 gradients vs autograd", [gw1, gb1, gw2, gb2], [p.grad.item() for p in P])
for k, v in dict(z1=z1, h1=h1, yh=yh, L=L, d2=d2, gw2=gw2, d1=d1, gw1=gw1, gb1=gb1).items():
    out.val(f"p13_{k}", v, 4)
w1n, w2n, b1n, b2n = w1 - eta * gw1, w2 - eta * gw2, b1 - eta * gb1, b2 - eta * gb2
out.val("p13_w1n", w1n, 3); out.val("p13_w2n", w2n, 3); out.val("p13_b1n", b1n, 3); out.val("p13_b2n", b2n, 3)
out.val("p13_Ln", 0.5 * (w2n * max(w1n * x13 + b1n, 0) + b2n - y13) ** 2, 4)
x14, w14, y14 = np.array([1.0, 2.0]), np.array([0.1, -0.2]), 1.0
p14 = sig(w14 @ x14); g14 = (p14 - y14) * x14
wt = T(w14, requires_grad=True); F.binary_cross_entropy(torch.sigmoid(wt @ T(x14)), T(y14)).backward()
out.check("P14 gradient", g14, wt.grad.numpy())
out.val("p14_p", p14, 4); out.val("p14_g", fv(g14, 4))
g15 = softmax(np.array([1.0, 2.0, 3.0])); g15[0] -= 1
out.val("p15_g", fv(g15, 4))
out.val("p17", sci(0.25 ** 6, 2))
h18 = 0.01; num18 = ((2 + h18) ** 3 - (2 - h18) ** 3) / (2 * h18)
out.val("p18_num", num18, 4); out.val("p18_fwd", ((2 + h18) ** 3 - 8) / h18, 4)

# --------------------------------------------------------------- optimisers
p = T([0.0], requires_grad=True); opt = torch.optim.SGD([p], lr=0.1, momentum=0.9); traj = []
for _ in range(3):
    opt.zero_grad(); p.grad = T([1.0]); opt.step(); traj.append(p.item())
v_, th_ = 0.0, 0.0; ours = []
for _ in range(3):
    v_ = 0.9 * v_ + 1.0; th_ -= 0.1 * v_; ours.append(th_)
out.check("P19 momentum steps vs torch", ours, traj)
out.val("p19", fv(ours, 2))
p = T([1.0], requires_grad=True); opt = torch.optim.Adam([p], lr=0.01); tr20 = []
for gg in (0.2, -0.1):
    opt.zero_grad(); p.grad = T([gg]); opt.step(); tr20.append(p.item())
m1 = 0.1 * 0.2; v1 = 0.001 * 0.04; m2 = 0.9 * m1 + 0.1 * (-0.1); v2 = 0.999 * v1 + 0.001 * 0.01
mh2, vh2 = m2 / (1 - 0.81), v2 / (1 - 0.999 ** 2)
th2 = 1 - 0.01 - 0.01 * mh2 / (math.sqrt(vh2) + 1e-8)
out.check("P20 Adam two steps vs torch", th2, tr20[1])
out.val("p20_th1", tr20[0], 4); out.val("p20_m2", m2, 4); out.val("p20_v2", sci(v2, 3)); out.val("p20_mh2", mh2, 4)
out.val("p20_vh2", sci(vh2, 3)); out.val("p20_th2", th2, 5)
p = T([0.0], requires_grad=True); opt = torch.optim.RMSprop([p], lr=0.01, alpha=0.9, eps=0.0)
opt.zero_grad(); p.grad = T([2.0]); opt.step()
out.check("P21 RMSProp first step", -0.01 * 2 / math.sqrt(0.1 * 4), p.item())
out.val("p21_v", 0.1 * 4, 1); out.val("p21_step", 0.01 * 2 / math.sqrt(0.4), 4)
out.val("p22_it", math.ceil(45000 / 64)); out.val("p22_tot", thousands(20 * math.ceil(45000 / 64)))
out.val("p23", 0.5 * 0.2 * (1 + math.cos(math.pi * 30 / 120)), 4)
out.val("p24", sci(3e-4 * 100 / 500, 1))
g25 = np.array([1.0, 2.0, 2.0]); t25 = T(g25.copy()); pz = torch.zeros(3, requires_grad=True); pz.grad = t25
torch.nn.utils.clip_grad_norm_([pz], 1.5)
out.check("P25 clip", g25 * 1.5 / 3, pz.grad.numpy())
out.val("p25", fv(g25 * 0.5, 1))
lmin, lmax = 0.5, 8.0
out.val("p26_max", 2 / lmax, 3); out.val("p26_opt", 2 / (lmin + lmax), 4); out.val("p26_rate", (lmax / lmin - 1) / (lmax / lmin + 1), 4)

# ----------------------------------------------------------- init / norm / reg
out.val("p27_he", math.sqrt(2 / 512), 4); out.val("p27_xav", math.sqrt(6 / 400), 4)
xb = np.array([2.0, 4.0, 6.0, 8.0]); bnm = nn.BatchNorm1d(1)
with torch.no_grad():
    bnm.weight.fill_(2.0); bnm.bias.fill_(1.0)
ybn = bnm(T(xb)[:, None])[:, 0].detach().numpy()
ours28 = 2 * (xb - 5) / math.sqrt(xb.var() + 1e-5) + 1
out.check("P28 BN vs torch", ours28, ybn)
out.val("p28_var", xb.var(), 1); out.val("p28_y", fv(ours28, 3))
out.val("p29_p", npar(nn.BatchNorm2d(128))); out.val("p29_b", 2 * 128)
x30 = np.array([2.0, 4.0, 6.0]); ln30 = F.layer_norm(T(x30), (3,)).numpy()
out.check("P30 LN", (x30 - 4) / math.sqrt(x30.var() + 1e-5), ln30)
out.val("p30", fv(ln30, 4))
out.val("p31_scale", 1 / 0.75, 4)
w32 = 2.0 - 0.1 * (0.5 + 2 * 0.05 * 2.0)
out.val("p32", w32, 3)

# ---------------------------------------------------------------------- CNNs
layers = [nn.Conv2d(3, 64, 7, 2, 3), nn.MaxPool2d(3, 2, 1), nn.Conv2d(64, 128, 3, 1, 1), nn.Conv2d(128, 256, 3, 2, 1)]
xx = torch.zeros(1, 3, 128, 128); shp, prm = [], []
for l in layers:
    xx = l(xx); shp.append(f"{xx.shape[2]}\\times{xx.shape[3]}\\times{xx.shape[1]}"); prm.append(npar(l))
out.check("P35 params formula", prm, [(49 * 3 + 1) * 64, 0, (9 * 64 + 1) * 128, (9 * 128 + 1) * 256], atol=0)
for i in range(4):
    out.val(f"p35_s{i}", shp[i]); out.val(f"p35_p{i}", thousands(prm[i]))
out.val("p35_tot", thousands(sum(prm)))
o36 = (32 - 3 * 2 - 1) // 1 + 1
out.check("P36 dilated size", o36, nn.Conv2d(1, 1, 3, dilation=3)(torch.zeros(1, 1, 32, 32)).shape[-1], atol=0)
out.val("p36", o36)
c37 = nn.Conv2d(32, 64, 5)
out.check("P37 params", npar(c37), (25 * 32 + 1) * 64, atol=0)
out.val("p37_p", thousands(npar(c37))); out.val("p37_mac", human(25 * 32 * 64 * 28 * 28))
def rf(ls):
    r, j = 1, 1
    for k, s in ls:
        r += (k - 1) * j; j *= s
    return r
out.val("p38", rf([(3, 1)] * 4 + [(2, 2), (3, 1)]))
out.val("p39_fc", thousands((32 * 32 * 3 + 1) * 10)); out.val("p39_conv", (27 + 1) * 10)
n40 = npar(nn.Conv2d(256, 256, 3, groups=256, bias=False)) + npar(nn.Conv2d(256, 512, 1, bias=False))
out.check("P40 depthwise separable", n40, 9 * 256 + 256 * 512, atol=0)
out.val("p40", thousands(n40)); out.val("p40_std", thousands(9 * 256 * 512)); out.val("p40_ratio", 9 * 256 * 512 / n40, 1)
out.val("p41", thousands(512 * 1000 + 1000)); out.val("p41_fc", human(7 * 7 * 512 * 1000 + 1000))
o42 = nn.ConvTranspose2d(1, 1, 4, 2, 1)(torch.zeros(1, 1, 7, 7)).shape[-1]
out.check("P42 transposed", o42, (7 - 1) * 2 - 2 + 4, atol=0); out.val("p42", o42)
out.val("p43", thousands(npar(nn.Conv2d(512, 128, 1))))
A_, B_ = (0, 0, 4, 6), (2, 3, 6, 9)
inter = max(0, min(A_[2], B_[2]) - max(A_[0], B_[0])) * max(0, min(A_[3], B_[3]) - max(A_[1], B_[1]))
union = 24 + 24 - inter
out.val("p44_i", inter); out.val("p44_u", union); out.val("p44", inter / union, 4)

# ---------------------------------------------------------------------- RNNs
out.check("P45 LSTM", npar(nn.LSTM(32, 64)), 4 * (64 * 96 + 128), atol=0)
out.val("p45_lstm", thousands(npar(nn.LSTM(32, 64)))); out.val("p45_gru", thousands(npar(nn.GRU(32, 64))))
out.val("p45_blk", thousands(64 * 96 + 128))
n46 = npar(nn.LSTM(16, 32, num_layers=2, bidirectional=True))
out.check("P46", n46, 2 * 4 * (32 * 48 + 64) + 2 * 4 * (32 * 96 + 64), atol=0)
out.val("p46", thousands(n46)); out.val("p46_l1", thousands(2 * 4 * (32 * 48 + 64))); out.val("p46_l2", thousands(2 * 4 * (32 * 96 + 64)))
h = 0.0; hs = []
for xt in (1.0, 0.0, 0.0):
    h = math.tanh(xt + 0.5 * h); hs.append(h)
out.val("p47", fv(hs, 4))
P0 = {"a": 0.6, "b": 0.4}; Pn = {"a": {"a": 0.55, "b": 0.45}, "b": {"a": 0.9, "b": 0.1}}
seqs = {s1 + s2: P0[s1] * Pn[s1][s2] for s1 in "ab" for s2 in "ab"}
greedy = "a" + max(Pn["a"], key=Pn["a"].get)
out.val("p48_greedy", greedy); out.val("p48_gp", seqs[greedy], 2)
best = max(seqs, key=seqs.get); out.val("p48_best", best); out.val("p48_bp", seqs[best], 2)
l49 = nn.LSTM(10, 20, batch_first=True, bidirectional=True)
o49, (h49, _) = l49(torch.zeros(8, 12, 10))
out.val("p49_o", ",".join(map(str, o49.shape))); out.val("p49_h", ",".join(map(str, h49.shape)))

# ----------------------------------------------------------------- attention
Q = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 1.0]]); V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
S = Q @ Q.T / math.sqrt(2); A = np.apply_along_axis(softmax, 1, S); O = A @ V
Ot = F.scaled_dot_product_attention(T(Q)[None], T(Q)[None], T(V)[None])[0].numpy()
out.check("P50 attention vs torch", O, Ot)
out.val("p50_s", fv(S[0], 3)); out.val("p50_a", fv(A[0], 3)); out.val("p50_o", fv(O[0], 3))
out.check("P51 MHA", npar(nn.MultiheadAttention(384, 6)), 4 * 384 ** 2 + 4 * 384, atol=0)
out.val("p51", thousands(4 * 384 ** 2 + 4 * 384)); out.val("p51_dk", 384 // 6)
n52 = npar(nn.TransformerEncoderLayer(256, 4, 1024))
out.check("P52 encoder layer", n52, (4 * 256 ** 2 + 4 * 256) + (2 * 256 * 1024 + 1024 + 256) + 4 * 256, atol=0)
out.val("p52", thousands(n52)); out.val("p52_att", thousands(4 * 256 ** 2 + 4 * 256)); out.val("p52_ffn", thousands(2 * 256 * 1024 + 1024 + 256))
emb53 = 32000 * 512 + 512 * 512
layer53 = npar(nn.TransformerEncoderLayer(512, 8, 2048))
tot53 = emb53 + 6 * layer53 + 2 * 512
out.val("p53_emb", human(emb53)); out.val("p53_layer", thousands(layer53)); out.val("p53_tot", human(tot53))
out.val("p54", f"{2048 * 2048 * 16 * 2 / 2**20:.0f}")
out.val("p55", f"{2 * 12 * 12 * 64 * 1024 * 4 * 2 / 2**20:.0f}")
out.val("p56", 4 * 3 // 2)
pe = [math.sin(2), math.cos(2), math.sin(2 / 100), math.cos(2 / 100)]
out.val("p57", fv(pe, 4))

# ---------------------------------------------------------------------- LLMs
out.val("p58_lora", human(24 * 4 * 4 * 2 * 2048)); out.val("p58_full", human(24 * 4 * 2048 * 2048))
out.val("p58_pct", 100 * (4 * 2 * 2048) / 2048 ** 2, 2)
out.val("p59", math.exp(2.0), 2)
pp = np.array([0.5, 0.2, 0.15, 0.1, 0.05]); cum = np.cumsum(pp)
nn60 = int(np.searchsorted(cum, 0.8) + 1)
out.val("p60_n", nn60); out.val("p60", fv(pp[:nn60] / pp[:nn60].sum(), 3)); out.val("p60_cum", fv(cum, 2))
out.val("p61_16", 13 * 2); out.val("p61_8", 13); out.val("p61_4", 6.5, 1)
mu62, s62 = np.array([0.5, -0.5]), np.array([1.0, 0.5])
kl62 = np.sum(0.5 * (mu62 ** 2 + s62 ** 2 - 1 - np.log(s62 ** 2)))
ref = torch.distributions.kl_divergence(torch.distributions.Normal(T(mu62), T(s62)), torch.distributions.Normal(0.0, 1.0)).sum().item()
out.check("P62 KL", kl62, ref)
out.val("p62", kl62, 4); out.val("p62_a", 0.5 * (0.25 + 1 - 1 - 0), 4); out.val("p62_b", 0.5 * (0.25 + 0.25 - 1 - math.log(0.25)), 4)
