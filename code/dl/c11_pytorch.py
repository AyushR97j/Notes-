"""Chapter 11: PyTorch practice. Broadcasting, the canonical loop, a catalogue of
silent bugs each demonstrated with numbers, overfitting one batch, gradient
accumulation == large batch, checkpoint/resume == uninterrupted, reproducibility,
mixed-precision ranges, training-memory arithmetic, and one-page GNN / RL /
time-series examples."""
import copy
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from common import setup
from dlutil import sci

out = setup(__file__)
rng = np.random.default_rng(0)

# ------------------------------------------------------------- broadcasting
a = torch.zeros(3, 1); b = torch.zeros(1, 4)
out.val("bc1", "\\times".join(map(str, (a + b).shape)))
try:
    torch.zeros(2, 3) + torch.zeros(2)
    out.val("bc_err", "no error")
except RuntimeError:
    out.val("bc_err", "RuntimeError")
out.val("bc3", "\\times".join(map(str, (torch.zeros(2, 3) + torch.zeros(3)).shape)))
# the (B,1) - (B,) bug
pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]]); y = torch.tensor([1.0, 2.0, 3.0, 4.0])
wrong = ((pred - y) ** 2).mean().item()
right = ((pred.squeeze(1) - y) ** 2).mean().item()
out.val("bc_wrong", wrong, 3); out.val("bc_right", right, 1)
out.val("bc_shape", "\\times".join(map(str, (pred - y).shape)))

# --------------------------------------------------------- data and a loop
X, Y = make_classification(1200, 20, n_informative=8, random_state=0)
X = torch.tensor(X, dtype=torch.float32); Y = torch.tensor(Y)
tr = TensorDataset(X[:1000], Y[:1000]); va = (X[1000:], Y[1000:])

def make_model(p=0.3):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 2))

# [[loop]]
def train(model, loader, epochs, lr=1e-2, loss_fn=nn.CrossEntropyLoss()):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()                              # dropout on, BN uses batch statistics
        for xb, yb in loader:
            opt.zero_grad()                        # gradients accumulate otherwise
            loss = loss_fn(model(xb), yb)          # raw logits in, class indices as targets
            loss.backward()
            opt.step()
    return model

@torch.no_grad()                                   # no graph, less memory
def evaluate(model, X, Y):
    model.eval()                                   # dropout off, BN uses running stats
    return (model(X).argmax(1) == Y).float().mean().item()
# [[/loop]]
loader = DataLoader(tr, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))
m = train(make_model(), loader, 10)
acc_eval = evaluate(m, *va)
m.train()
with torch.no_grad():
    acc_train_mode = (m(va[0]).argmax(1) == va[1]).float().mean().item()
    o1 = m(va[0]); o2 = m(va[0])
out.val("acc_eval", 100 * acc_eval, 1); out.val("acc_trainmode", 100 * acc_train_mode, 1)
out.val("nondet", (o1 - o2).abs().max().item(), 3)
# the same model in eval mode is deterministic
m.eval()
with torch.no_grad():
    out.check("eval mode is deterministic", (m(va[0]) - m(va[0])).abs().max().item(), 0.0)
# single-example prediction in train mode with BN
m.train()
try:
    m(va[0][:1]); out.val("bn1", "ran")
except ValueError:
    out.val("bn1", "ValueError")

# softmax before CrossEntropyLoss
class SoftmaxThenCE(nn.Module):
    def forward(self, logits, y):
        return F.cross_entropy(torch.softmax(logits, 1), y)
m_ok = train(make_model(0.0), loader, 10)
m_bad = train(make_model(0.0), loader, 10, loss_fn=SoftmaxThenCE())
with torch.no_grad():
    m_ok.eval(); m_bad.eval()
    l_ok = F.cross_entropy(m_ok(X[:1000]), Y[:1000]).item()
    l_bad_reported = SoftmaxThenCE()(m_bad(X[:1000]), Y[:1000]).item()
out.val("sm_ok_acc", 100 * evaluate(m_ok, *va), 1); out.val("sm_bad_acc", 100 * evaluate(m_bad, *va), 1)
out.val("sm_ok_loss", l_ok, 3); out.val("sm_bad_loss", l_bad_reported, 3)
out.val("sm_floor", F.cross_entropy(torch.tensor([[1.0, 0.0]]), torch.tensor([0])).item(), 3)

# wrong softmax dim
z = torch.randn(4, 3)
out.val("dim_rowsum", ", ".join(f"{v:.2f}" for v in torch.softmax(z, dim=0).sum(1).tolist()))

# forgetting zero_grad
torch.manual_seed(0)
lin = nn.Linear(3, 1); xg = torch.randn(5, 3)
lin(xg).sum().backward(); g1 = lin.weight.grad.clone()
lin(xg).sum().backward(); g2 = lin.weight.grad.clone()
out.check("without zero_grad the second backward doubles the gradient", g2.numpy(), 2 * g1.numpy())

# unshuffled, label-sorted data
order = torch.argsort(Y[:1000])
sorted_ds = TensorDataset(X[:1000][order], Y[:1000][order])
m_sorted = train(make_model(0.0), DataLoader(sorted_ds, batch_size=32, shuffle=False), 1, lr=1e-2)
m_shuf = train(make_model(0.0), DataLoader(sorted_ds, batch_size=32, shuffle=True,
                                          generator=torch.Generator().manual_seed(0)), 1, lr=1e-2)
out.val("sorted_acc", 100 * evaluate(m_sorted, *va), 1); out.val("shuf_acc", 100 * evaluate(m_shuf, *va), 1)

# label leakage
Xn = X.numpy(); Yn = Y.numpy()
leak = np.c_[Xn, Yn + 0.1 * rng.normal(size=len(Yn))]
lr_leak = LogisticRegression(max_iter=2000).fit(leak[:1000], Yn[:1000])
lr_ok = LogisticRegression(max_iter=2000).fit(Xn[:1000], Yn[:1000])
out.val("leak_val", 100 * lr_leak.score(leak[1000:], Yn[1000:]), 1); out.val("noleak_val", 100 * lr_ok.score(Xn[1000:], Yn[1000:]), 1)

# --------------------------------------------------------- overfit one batch
xb, yb = next(iter(loader))
mo = make_model(0.0); opt = torch.optim.Adam(mo.parameters(), lr=1e-2)
for step in range(200):
    mo.train(); opt.zero_grad(); l = F.cross_entropy(mo(xb), yb); l.backward(); opt.step()
out.val("one_batch_loss", sci(l.item(), 1)); out.val("one_batch_n", len(yb))

# -------------------------------------------- gradient accumulation == big batch
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(20, 16), nn.Tanh(), nn.Linear(16, 2))
xb32, yb32 = X[:32], Y[:32]
net.zero_grad(); F.cross_entropy(net(xb32), yb32).backward()
g_big = [p.grad.clone() for p in net.parameters()]
net.zero_grad()
for k in range(4):
    (F.cross_entropy(net(xb32[8 * k:8 * k + 8]), yb32[8 * k:8 * k + 8]) / 4).backward()
g_acc = [p.grad.clone() for p in net.parameters()]
out.check("accumulating 4 micro-batches of 8 (loss/4) == one batch of 32",
          torch.cat([g.flatten() for g in g_acc]).numpy(), torch.cat([g.flatten() for g in g_big]).numpy(), atol=1e-6)

# ---------------------------------------- checkpoint / resume == uninterrupted
def run(steps, model, opt, g):
    for _ in range(steps):
        idx = torch.randint(0, 1000, (32,), generator=g)
        opt.zero_grad(); F.cross_entropy(model(X[idx]), Y[idx]).backward(); opt.step()
torch.manual_seed(0)
mA = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 2)); oA = torch.optim.Adam(mA.parameters(), 1e-2)
gA = torch.Generator().manual_seed(5); run(20, mA, oA, gA)
# [[ckpt]]
buf = io.BytesIO()
torch.save({"model": mA.state_dict(), "opt": oA.state_dict(), "rng": gA.get_state()}, buf)
buf.seek(0); ck = torch.load(buf)
mB = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 2)); mB.load_state_dict(ck["model"])
oB = torch.optim.Adam(mB.parameters(), 1e-2); oB.load_state_dict(ck["opt"])
gB = torch.Generator(); gB.set_state(ck["rng"])
# [[/ckpt]]
run(20, mA, oA, gA); run(20, mB, oB, gB)
out.check("resume from checkpoint == uninterrupted training",
          torch.cat([p.flatten() for p in mB.parameters()]).detach().numpy(),
          torch.cat([p.flatten() for p in mA.parameters()]).detach().numpy(), atol=0)
# forgetting the optimiser state
mC = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 2)); mC.load_state_dict(ck["model"])
oC = torch.optim.Adam(mC.parameters(), 1e-2); gC = torch.Generator(); gC.set_state(ck["rng"]); run(20, mC, oC, gC)
diff = (torch.cat([p.flatten() for p in mC.parameters()]) - torch.cat([p.flatten() for p in mA.parameters()])).abs().max()
out.val("ckpt_noopt_diff", diff.item(), 3)

# -------------------------------------------------------------- reproducibility
def init_weights(seed):
    torch.manual_seed(seed); return nn.Linear(5, 5).weight.detach().clone()
out.check("same seed -> identical initial weights", init_weights(7).numpy(), init_weights(7).numpy(), atol=0)
out.val("diffseed", (init_weights(7) - init_weights(8)).abs().max().item(), 3)

# ----------------------------------------------------------- precision ranges
for nm, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
    fi = torch.finfo(dt)
    out.val(f"{nm}_max", sci(fi.max, 2)); out.val(f"{nm}_eps", sci(fi.eps, 2)); out.val(f"{nm}_tiny", sci(fi.tiny, 2))
g_small = torch.tensor(1e-8)
out.val("fp16_underflow", g_small.half().item()); out.val("fp16_scaled", sci(float((g_small * 65536).half().float() / 65536), 2))
A_ = torch.randn(256, 256); B_ = torch.randn(256, 256)
ref = A_.double() @ B_.double()
err16 = ((A_.bfloat16() @ B_.bfloat16()).double() - ref).abs().max().item()
out.val("bf16_mm_err", err16, 2); out.val("fp32_mm_err", sci(((A_ @ B_).double() - ref).abs().max().item(), 1))

# ------------------------------------------------------- training memory
out.val("mem_bytes", 16); out.val("mem_7b", 7 * 16)

# ------------------------------------------------------------------ GNN
Aadj = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], float)
H0 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
Wg = np.array([[1.0, 0.5], [-0.5, 1.0]])
# [[gcn]]
def gcn_layer(A, H, W):
    A_hat = A + np.eye(len(A))                       # add self-loops
    d = A_hat.sum(1)
    A_norm = A_hat / np.sqrt(d[:, None] * d[None, :])  # D^-1/2 (A+I) D^-1/2
    return np.maximum(0, A_norm @ H @ W)             # aggregate neighbours, transform, ReLU
# [[/gcn]]
H1 = gcn_layer(Aadj, H0, Wg)
P = np.eye(4)[[2, 0, 3, 1]]
out.check("GCN layer is permutation-equivariant", gcn_layer(P @ Aadj @ P.T, P @ H0, Wg), P @ H1)
# message-passing form: sum over neighbours j of h_j / sqrt(d_i d_j)
d_ = Aadj.sum(1) + 1
mp = np.array([sum(H0[j] / np.sqrt(d_[i] * d_[j]) for j in range(4) if Aadj[i, j] or i == j) for i in range(4)])
out.check("GCN matrix form == per-node message passing", H1, np.maximum(0, mp @ Wg))
out.tex("gcn_h1", " \\\\ ".join(" & ".join(f"{v:.3f}" for v in row) for row in H1))

# ------------------------------------------------------------- RL: bandit
true_p = np.array([0.2, 0.5, 0.8])
theta = np.zeros(3); lr = 0.1; g = np.random.default_rng(1)
for t in range(2000):
    pi = np.exp(theta) / np.exp(theta).sum()
    a_ = g.choice(3, p=pi)
    r = float(g.random() < true_p[a_])
    grad_logpi = -pi; grad_logpi[a_] += 1          # d log pi(a) / d theta
    theta += lr * r * grad_logpi                    # REINFORCE (no baseline)
pi = np.exp(theta) / np.exp(theta).sum()
out.val("rl_pi", ", ".join(f"{v:.3f}" for v in pi))
# DQN target
r_, gamma_, qn = 1.0, 0.9, np.array([2.0, 5.0, 3.0])
out.val("dqn_target", r_ + gamma_ * qn.max(), 1)

# --------------------------------------------------------- time-series windows
series = np.arange(100.0)
W, Hh = 24, 6
nwin = len(series) - W - Hh + 1
out.val("ts_nwin", nwin); out.val("ts_W", W); out.val("ts_H", Hh)
out.val("p_13b", 1.3 * 16, 1)
