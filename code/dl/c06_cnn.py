"""Chapter 6: convolutions. Cross-correlation vs convolution by hand (vs torch and
scipy), output-shape formula (many cases vs torch), parameter/MAC counts of a
small CNN, receptive fields (formula vs gradient probe), pooling, 1x1 and
depthwise-separable convs, transposed conv, block counts for VGG/ResNet-style
configs, IoU and NMS."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import correlate2d, convolve2d
from common import setup, thousands
from dlutil import human

out = setup(__file__)
torch.set_default_dtype(torch.float64)

# ----------------------------------------------------- conv by hand (2-D)
X = np.array([[1, 2, 0, 1], [0, 1, 3, 2], [2, 1, 0, 1], [1, 0, 2, 3]], float)
K = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)     # a Sobel-like kernel
# [[xcorr]]
def conv2d_single(X, K, stride=1, pad=0):
    """What deep-learning 'convolution' computes: cross-correlation (no flip)."""
    X = np.pad(X, pad)
    kh, kw = K.shape
    H = (X.shape[0] - kh) // stride + 1
    W = (X.shape[1] - kw) // stride + 1
    Y = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            patch = X[i * stride:i * stride + kh, j * stride:j * stride + kw]
            Y[i, j] = np.sum(patch * K)                 # elementwise product, then sum
    return Y
# [[/xcorr]]
Y = conv2d_single(X, K)
Yt = F.conv2d(torch.tensor(X)[None, None], torch.tensor(K)[None, None])[0, 0].numpy()
out.check("hand cross-correlation vs torch conv2d", Y, Yt)
out.check("hand cross-correlation vs scipy correlate2d", Y, correlate2d(X, K, mode="valid"))
Yc = convolve2d(X, K, mode="valid")
out.check("true convolution = cross-correlation with flipped kernel", Yc, conv2d_single(X, K[::-1, ::-1]))
for i in range(2):
    for j in range(2):
        out.val(f"y{i}{j}", Y[i, j], 0); out.val(f"yc{i}{j}", Yc[i, j], 0)
P = X[0:3, 0:3]
out.val("y00_terms", " + ".join(f"({int(a)})({int(b)})" for a, b in zip(P.ravel(), K.ravel()) if b != 0))
Ys2 = conv2d_single(X, K, stride=1, pad=1)
out.check("padding 1 vs torch", Ys2, F.conv2d(torch.tensor(X)[None, None], torch.tensor(K)[None, None], padding=1)[0, 0].numpy())
out.val("pad_shape", f"{Ys2.shape[0]}\\times{Ys2.shape[1]}")

# --------------------------------------------------- output-shape formula
# [[outshape]]
def conv_out(n, k, s=1, p=0, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1
# [[/outshape]]
cases = [(32, 5, 1, 0, 1), (32, 3, 1, 1, 1), (224, 7, 2, 3, 1), (227, 11, 4, 0, 1), (28, 3, 2, 1, 1),
         (64, 3, 1, 2, 2), (7, 3, 2, 0, 1), (10, 4, 3, 1, 1), (56, 1, 1, 0, 1), (112, 3, 2, 1, 1)]
rows = []
for n, k, s, p, d in cases:
    o = conv_out(n, k, s, p, d)
    ot = nn.Conv2d(1, 1, k, s, p, d)(torch.zeros(1, 1, n, n)).shape[-1]
    out.check(f"out size n={n} k={k} s={s} p={p} d={d}", o, ot, atol=0)
    rows.append(f"${n}$ & ${k}$ & ${s}$ & ${p}$ & ${d}$ & $\\lfloor({n}{'+' + str(2*p) if p else ''}"
                f"-{d*(k-1)+1})/{s}\\rfloor+1$ & ${o}$ \\\\")
out.tex("shape_rows", "\n".join(rows))
# the classic OA question
c = nn.Conv2d(3, 10, 5)
out.check("32x32x3 -> 5x5x3 x10 params", sum(q.numel() for q in c.parameters()), (5 * 5 * 3 + 1) * 10, atol=0)
out.val("oa_params", (5 * 5 * 3 + 1) * 10)
out.val("oa_macs", thousands(5 * 5 * 3 * 10 * 28 * 28))

# -------------------------------------------- a small CNN: shapes, params, MACs
# [[smallcnn]]
net = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),          # 32x32x16
    nn.MaxPool2d(2),                                    # 16x16x16
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),         # 16x16x32
    nn.MaxPool2d(2),                                    # 8x8x32
    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 4x4x64
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),              # 64
    nn.Linear(64, 10))
# [[/smallcnn]]
x = torch.zeros(1, 3, 32, 32)
rows, tot_p, tot_m = [], 0, 0
h, w, cch = 32, 32, 3
for layer in net:
    y = layer(x)
    name = type(layer).__name__
    if isinstance(layer, nn.Conv2d):
        k = layer.kernel_size[0]; s = layer.stride[0]; p = layer.padding[0]
        ho = conv_out(h, k, s, p); co = layer.out_channels
        prm = (k * k * cch + 1) * co; mac = k * k * cch * co * ho * ho
        desc = f"conv {k}$\\times${k}, {co}" + (f", stride {s}" if s > 1 else "")
    elif isinstance(layer, nn.MaxPool2d):
        ho, co, prm, mac = h // 2, cch, 0, 0; desc = "max-pool 2$\\times$2"
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        ho, co, prm, mac = 1, cch, 0, 0; desc = "global avg pool"
    elif isinstance(layer, nn.Linear):
        ho, co, prm, mac = 1, layer.out_features, (layer.in_features + 1) * layer.out_features, layer.in_features * layer.out_features
        desc = f"linear {layer.in_features}$\\to${layer.out_features}"
    else:
        x = y; continue
    tp = sum(q.numel() for q in layer.parameters())
    out.check(f"{name} params formula vs torch", prm, tp, atol=0)
    if y.dim() == 4:
        out.check(f"{name} output shape", [co, ho, ho], list(y.shape[1:]), atol=0)
        shape = f"${ho}\\times{ho}\\times{co}$"
    else:
        shape = f"${y.shape[1]}$"
    rows.append(f"{desc} & {shape} & ${thousands(prm)}$ & ${thousands(mac)}$ \\\\")
    tot_p += prm; tot_m += mac
    h, w, cch = ho, ho, co
    x = y
out.check("small CNN total params vs torch", tot_p, sum(q.numel() for q in net.parameters()), atol=0)
out.tex("cnn_rows", "\n".join(rows))
out.val("cnn_params", thousands(tot_p)); out.val("cnn_macs", thousands(tot_m))
fc_equiv = (32 * 32 * 3 + 1) * (32 * 32 * 16)
out.val("fc_equiv", thousands(fc_equiv)); out.val("conv1_params", (3 * 3 * 3 + 1) * 16)
out.val("fc_ratio", thousands(round(fc_equiv / ((3 * 3 * 3 + 1) * 16))))
out.val("fc_224", human((224 * 224 * 3) * 1000 + 1000))

# ----------------------------------------------------------- receptive field
# [[rf]]
def receptive_field(layers):                  # layers: list of (kernel, stride, dilation)
    r, j = 1, 1                               # receptive field, jump (input pixels per step)
    for k, s, d in layers:
        r += (k - 1) * d * j
        j *= s
    return r
# [[/rf]]
def rf_probe(layers, n=64):
    mods = [nn.Conv2d(1, 1, k, s, padding=0, dilation=d, bias=False) for k, s, d in layers]
    for m in mods:
        nn.init.constant_(m.weight, 1.0)
    xin = torch.ones(1, 1, n, n, requires_grad=True)
    y = xin
    for m in mods:
        y = m(y)
    c = y.shape[-1] // 2
    y[0, 0, c, c].backward()
    nz = (xin.grad[0, 0].abs() > 0).any(0).nonzero()
    return int(nz.max() - nz.min() + 1)
rf_cases = {
    "two3": [(3, 1, 1)] * 2, "three3": [(3, 1, 1)] * 3, "pool": [(3, 1, 1), (2, 2, 1), (3, 1, 1)],
    "s2": [(3, 2, 1), (3, 2, 1), (3, 2, 1)], "dil": [(3, 1, 1), (3, 1, 2), (3, 1, 4)],
    "vggblock": [(3, 1, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)],
}
for kname, ls in rf_cases.items():
    r = receptive_field(ls)
    out.check(f"receptive field {kname}: formula vs gradient probe", r, rf_probe(ls), atol=0)
    out.val(f"rf_{kname}", r)
out.val("rf_par_two3", 2 * 9); out.val("rf_par_5", 25)
out.val("rf_par_three3", 3 * 9); out.val("rf_par_7", 49)

# -------------------------------------------------------------------- pooling
Xp = np.array([[1, 3, 2, 1], [4, 2, 0, 1], [5, 1, 2, 7], [0, 2, 3, 4]], float)
mp = Xp.reshape(2, 2, 2, 2).max(axis=(1, 3)); ap = Xp.reshape(2, 2, 2, 2).mean(axis=(1, 3))
xt = torch.tensor(Xp)[None, None].requires_grad_()
mpt = F.max_pool2d(xt, 2); mpt.sum().backward()
out.check("max-pool 2x2 vs torch", mp, mpt[0, 0].detach().numpy())
out.check("avg-pool 2x2 vs torch", ap, F.avg_pool2d(torch.tensor(Xp)[None, None], 2)[0, 0].numpy())
for i in range(2):
    for j in range(2):
        out.val(f"mp{i}{j}", mp[i, j], 0); out.val(f"ap{i}{j}", ap[i, j], 2)
grad_mask = xt.grad[0, 0].numpy()
out.tex("mp_grad", " \\\\ ".join(" & ".join(f"{int(v)}" for v in row) for row in grad_mask))
out.val("pool_params", sum(q.numel() for q in nn.MaxPool2d(2).parameters()))

# --------------------------------------------- 1x1, depthwise separable, groups
cin, cout, k = 256, 256, 3
std = nn.Conv2d(cin, cout, k, padding=1, bias=False)
dw = nn.Conv2d(cin, cin, k, padding=1, groups=cin, bias=False)
pw = nn.Conv2d(cin, cout, 1, bias=False)
n_std = sum(q.numel() for q in std.parameters()); n_dw = sum(q.numel() for q in dw.parameters())
n_pw = sum(q.numel() for q in pw.parameters())
out.check("standard 3x3 params", n_std, k * k * cin * cout, atol=0)
out.check("depthwise params", n_dw, k * k * cin, atol=0)
out.check("pointwise params", n_pw, cin * cout, atol=0)
out.val("std_p", thousands(n_std)); out.val("dw_p", thousands(n_dw)); out.val("pw_p", thousands(n_pw))
out.val("sep_p", thousands(n_dw + n_pw)); out.val("sep_ratio", (n_dw + n_pw) / n_std, 3)
out.val("sep_ratio_formula", 1 / cout + 1 / k ** 2, 3)
xx = torch.randn(1, cin, 8, 8)
out.check("depthwise-separable output shape = standard", list(pw(dw(xx)).shape), list(std(xx).shape), atol=0)
g4 = nn.Conv2d(cin, cout, k, padding=1, groups=4, bias=False)
out.val("g4_p", thousands(sum(q.numel() for q in g4.parameters())))
bott = [nn.Conv2d(256, 64, 1, bias=False), nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.Conv2d(64, 256, 1, bias=False)]
n_bott = sum(q.numel() for m in bott for q in m.parameters())
out.val("bott_p", thousands(n_bott)); out.val("two33_p", thousands(2 * 9 * 256 * 256))
out.val("one11", 256 * 64)

# ---------------------------------------------------------- transposed conv
# [[tconv_shape]]
def tconv_out(n, k, s=1, p=0, d=1, op=0):
    return (n - 1) * s - 2 * p + d * (k - 1) + op + 1
# [[/tconv_shape]]
for n, kk, s, p in [(4, 4, 2, 1), (7, 3, 2, 1), (16, 2, 2, 0), (5, 3, 1, 1)]:
    tt = nn.ConvTranspose2d(1, 1, kk, s, p)(torch.zeros(1, 1, n, n)).shape[-1]
    out.check(f"transposed conv out n={n} k={kk} s={s} p={p}", tconv_out(n, kk, s, p), tt, atol=0)
out.val("tc_4_4_2_1", tconv_out(4, 4, 2, 1)); out.val("tc_16_2_2_0", tconv_out(16, 2, 2, 0))
# transposed conv on a 2x2 input by hand: scatter each input times the kernel
xi = np.array([[1.0, 2.0], [3.0, 4.0]]); kk = np.array([[1.0, 1.0], [1.0, 0.0]])
yo = np.zeros((3, 3))
for i in range(2):
    for j in range(2):
        yo[i:i + 2, j:j + 2] += xi[i, j] * kk
out.check("transposed conv by scattering vs torch", yo,
          F.conv_transpose2d(torch.tensor(xi)[None, None], torch.tensor(kk)[None, None])[0, 0].numpy())
out.tex("tc_rows", " \\\\ ".join(" & ".join(f"{int(v)}" for v in row) for row in yo))
# transposed conv is the gradient (adjoint) of conv
xa = torch.randn(1, 1, 5, 5); ka = torch.randn(1, 1, 3, 3); ya = torch.randn(1, 1, 3, 3)
lhs = (F.conv2d(xa, ka) * ya).sum(); rhs = (xa * F.conv_transpose2d(ya, ka)).sum()
out.check("<conv(x), y> = <x, conv_transpose(y)>", lhs.item(), rhs.item())

# ------------------------------------------------- VGG / ResNet configurations
def vgg16_like():
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
    layers, c = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2))
        else:
            layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU()]; c = v
    feats = nn.Sequential(*layers)
    head = nn.Sequential(nn.Flatten(), nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Linear(4096, 4096),
                         nn.ReLU(), nn.Linear(4096, 1000))
    return feats, head
feats, head = vgg16_like()
pf = sum(q.numel() for q in feats.parameters()); ph = sum(q.numel() for q in head.parameters())
out.check("VGG-16 config: feature map 7x7x512 at 224", list(feats(torch.zeros(1, 3, 224, 224)).shape), [1, 512, 7, 7], atol=0)
out.val("vgg_conv", human(pf)); out.val("vgg_fc", human(ph)); out.val("vgg_total", human(pf + ph))
out.val("vgg_fc_frac", 100 * ph / (pf + ph), 0)
out.val("vgg_fc1", human(512 * 7 * 7 * 4096 + 4096))

class Basic(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                   if stride != 1 or cin != cout else nn.Identity())
    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))
def resnet18_like():
    layers = [nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1)]
    c = 64
    for w, s in [(64, 1), (128, 2), (256, 2), (512, 2)]:
        layers += [Basic(c, w, s), Basic(w, w, 1)]; c = w
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1000)]
    return nn.Sequential(*layers)
r18 = resnet18_like()
out.val("r18_total", human(sum(q.numel() for q in r18.parameters())))
out.check("ResNet-18 config output shape", list(r18(torch.zeros(1, 3, 224, 224)).shape), [1, 1000], atol=0)
blk = Basic(64, 64, 1)
out.val("basic_p", thousands(sum(q.numel() for q in blk.parameters())))
out.val("basic_p_formula", thousands(2 * 9 * 64 * 64 + 4 * 64))

# --------------------------------------------------------------- IoU and NMS
# [[iou_nms]]
def iou(a, b):                                   # boxes as (x1, y1, x2, y2)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)

def nms(boxes, scores, thr=0.5):
    order = list(np.argsort(-scores))            # highest score first
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thr]
    return keep
# [[/iou_nms]]
A = (0, 0, 4, 4); Bx = (2, 2, 6, 6)
v = iou(A, Bx)
grid = np.zeros((8, 8), bool); ga = grid.copy(); gb = grid.copy()
ga[0:4, 0:4] = True; gb[2:6, 2:6] = True
out.check("IoU vs pixel counting", v, (ga & gb).sum() / (ga | gb).sum())
out.val("iou_ab", v, 4); out.val("iou_inter", 4); out.val("iou_union", 16 + 16 - 4)
boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 140, 150], [11, 9, 49, 51], [102, 98, 141, 148], [60, 10, 90, 40]], float)
scores = np.array([0.9, 0.8, 0.75, 0.7, 0.6, 0.3])
keep = nms(boxes, scores, 0.5)
# independent reference: vectorised IoU matrix + greedy suppression
def iou_matrix(b):
    x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ar[:, None] + ar[None, :] - inter)
M = iou_matrix(boxes); alive = np.ones(6, bool); ref = []
for i in np.argsort(-scores):
    if alive[i]:
        ref.append(i); alive &= ~(M[i] > 0.5); alive[i] = False
out.check("NMS vs vectorised reference", sorted(keep), sorted(ref), atol=0)
out.val("nms_keep", ", ".join(str(i + 1) for i in keep))
out.val("iou_12", M[0, 1], 3); out.val("iou_14", M[0, 3], 3); out.val("iou_35", M[2, 4], 3)

# ------------------------------------------------------------ problem values
pa = [nn.Conv2d(3, 32, 3, 1, 1), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, 2, 2)]
xx = torch.zeros(1, 3, 64, 64)
for i, m in enumerate(pa):
    xx = m(xx)
    out.val(f"pa_shape{i}", "\\times".join(str(v) for v in [xx.shape[2], xx.shape[3], xx.shape[1]]))
    out.val(f"pa_par{i}", thousands(sum(q.numel() for q in m.parameters())))
out.check("problem A conv2 params", sum(q.numel() for q in pa[2].parameters()), (25 * 32 + 1) * 64, atol=0)
out.val("pb_out", conv_out(7, 3, 3)); out.val("pb_unused", 7 + 7 - 1)
lsC = [(3, 1, 1), (3, 2, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1)]
out.check("problem C receptive field", receptive_field(lsC), rf_probe(lsC), atol=0)
out.val("pc_rf", receptive_field(lsC))
dwb = nn.Conv2d(128, 128, 3, padding=1, groups=128); pwb = nn.Conv2d(128, 256, 1); stb = nn.Conv2d(128, 256, 3, padding=1)
nd_, np_, ns_ = (sum(q.numel() for q in m.parameters()) for m in (dwb, pwb, stb))
out.val("pd_dw", thousands(nd_)); out.val("pd_pw", thousands(np_)); out.val("pd_tot", thousands(nd_ + np_))
out.val("pd_std", thousands(ns_)); out.val("pd_ratio", ns_ / (nd_ + np_), 1)
out.val("pf_iou", iou((0, 0, 10, 10), (5, 0, 15, 10)), 4)
