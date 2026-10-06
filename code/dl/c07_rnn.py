"""Chapter 7: recurrent networks. A scalar RNN unrolled by hand, parameter counts
(RNN/LSTM/GRU, stacked, bidirectional) vs torch, an LSTM and a GRU step from
scratch vs torch, gradient decay through time vs spectral radius, beam search vs
greedy vs brute force, CTC loss by enumerating alignments vs F.ctc_loss."""
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import setup, thousands
from dlutil import sci

out = setup(__file__)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

# ------------------------------------------------ scalar RNN unrolled by hand
wx, wh, b = 0.5, 0.8, 0.0
xs = [1.0, 0.5, -1.0]
h = 0.0; hs = []
for x in xs:
    h = math.tanh(wx * x + wh * h + b); hs.append(h)
rnn = nn.RNN(1, 1, batch_first=True)
with torch.no_grad():
    rnn.weight_ih_l0.fill_(wx); rnn.weight_hh_l0.fill_(wh); rnn.bias_ih_l0.zero_(); rnn.bias_hh_l0.zero_()
ht, _ = rnn(torch.tensor(xs)[None, :, None])
out.check("scalar RNN unrolled by hand vs nn.RNN", hs, ht[0, :, 0].detach().numpy())
for i, v in enumerate(hs):
    out.val(f"h{i+1}", v, 4)
out.val("pre2", wx * xs[1] + wh * hs[0], 4); out.val("pre3", wx * xs[2] + wh * hs[1], 4)
# gradient of h3 w.r.t. h1 through time
d32 = wh * (1 - hs[2] ** 2); d21 = wh * (1 - hs[1] ** 2)
out.val("dh3dh1", d32 * d21, 4)
h0 = torch.zeros(1, 1, 1)
x_t = torch.tensor(xs)
h1t = torch.tanh(wx * x_t[0] + torch.tensor(0.0)).requires_grad_()
h2t = torch.tanh(wx * x_t[1] + wh * h1t); h3t = torch.tanh(wx * x_t[2] + wh * h2t)
g, = torch.autograd.grad(h3t, h1t)
out.check("dh3/dh1 by chain rule vs autograd", d32 * d21, g.item())

# ------------------------------------------------------------- param counts
def n_params(m):
    return sum(p.numel() for p in m.parameters())
D, H = 10, 20
cnt = {
    "rnn": (nn.RNN(D, H), H * (D + H) + 2 * H),
    "lstm": (nn.LSTM(D, H), 4 * (H * (D + H) + 2 * H)),
    "gru": (nn.GRU(D, H), 3 * (H * (D + H) + 2 * H)),
    "lstm2": (nn.LSTM(D, H, num_layers=2), 4 * (H * (D + H) + 2 * H) + 4 * (H * (H + H) + 2 * H)),
    "bilstm": (nn.LSTM(D, H, bidirectional=True), 2 * 4 * (H * (D + H) + 2 * H)),
    "bilstm2": (nn.LSTM(D, H, num_layers=2, bidirectional=True),
                2 * 4 * (H * (D + H) + 2 * H) + 2 * 4 * (H * (2 * H + H) + 2 * H)),
}
for k, (m, formula) in cnt.items():
    out.check(f"{k} D={D} H={H} params", formula, n_params(m), atol=0)
    out.val(f"n_{k}", thousands(formula))
out.val("n_lstm_onebias", thousands(4 * (H * (D + H) + H)))
# OA-sized: LSTM input 100, hidden 128
D2, H2 = 100, 128
out.check("LSTM(100,128)", 4 * (H2 * (D2 + H2) + 2 * H2), n_params(nn.LSTM(D2, H2)), atol=0)
out.val("lstm_100_128", thousands(4 * (H2 * (D2 + H2) + 2 * H2)))
out.val("lstm_100_128_one", thousands(4 * (H2 * (D2 + H2) + H2)))
out.val("gru_100_128", thousands(3 * (H2 * (D2 + H2) + 2 * H2)))
out.check("GRU(100,128)", 3 * (H2 * (D2 + H2) + 2 * H2), n_params(nn.GRU(D2, H2)), atol=0)
out.val("rnn_100_128", thousands(H2 * (D2 + H2) + 2 * H2))
# output shapes
lstm = nn.LSTM(D, H, num_layers=2, bidirectional=True, batch_first=True)
o, (hn, cn) = lstm(torch.zeros(4, 7, D))
out.val("shape_o", ",".join(map(str, o.shape))); out.val("shape_h", ",".join(map(str, hn.shape)))

# ---------------------------------------------------- LSTM and GRU from scratch
sig = lambda z: 1 / (1 + np.exp(-z))
# [[lstm_cell]]
def lstm_cell(x, h, c, W, U, b):
    """W: (4H, D), U: (4H, H), b: (4H,), gates stacked in PyTorch order i, f, g, o."""
    z = W @ x + U @ h + b
    i, f, g, o = np.split(z, 4)
    i, f, o = sig(i), sig(f), sig(o)          # input, forget, output gates in (0, 1)
    g = np.tanh(g)                            # candidate cell update
    c = f * c + i * g                         # cell state: forget some, write some
    h = o * np.tanh(c)                        # hidden state: expose some of the cell
    return h, c
# [[/lstm_cell]]
# [[gru_cell]]
def gru_cell(x, h, W, U, bw, bu):
    """PyTorch gate order r, z, n; note r multiplies (U_n h + b_un)."""
    a, bb = W @ x + bw, U @ h + bu
    ar, az, an = np.split(a, 3); br, bz, bn = np.split(bb, 3)
    r, z = sig(ar + br), sig(az + bz)         # reset and update gates
    n = np.tanh(an + r * bn)                  # candidate
    return (1 - z) * n + z * h                # interpolate old and new
# [[/gru_cell]]
Dc, Hc = 3, 2
cell = nn.LSTMCell(Dc, Hc)
x = rng.normal(size=Dc); h0 = rng.normal(size=Hc); c0 = rng.normal(size=Hc)
W = cell.weight_ih.detach().numpy(); U = cell.weight_hh.detach().numpy()
bsum = (cell.bias_ih + cell.bias_hh).detach().numpy()
h1, c1 = lstm_cell(x, h0, c0, W, U, bsum)
ht, ct = cell(torch.tensor(x)[None], (torch.tensor(h0)[None], torch.tensor(c0)[None]))
out.check("LSTM cell h vs nn.LSTMCell", h1, ht[0].detach().numpy())
out.check("LSTM cell c vs nn.LSTMCell", c1, ct[0].detach().numpy())
gcell = nn.GRUCell(Dc, Hc)
hg = gru_cell(x, h0, gcell.weight_ih.detach().numpy(), gcell.weight_hh.detach().numpy(),
              gcell.bias_ih.detach().numpy(), gcell.bias_hh.detach().numpy())
out.check("GRU cell vs nn.GRUCell", hg, gcell(torch.tensor(x)[None], torch.tensor(h0)[None])[0].detach().numpy())

# a scalar LSTM step by hand (all weights given)
xs1, hp, cp = 1.0, 0.0, 0.5
wi, wf, wg, wo = (0.5, 0.0), (1.0, 0.0), (1.0, 0.0), (0.5, 0.0)   # (weight on x, bias)
iv = sig(wi[0] * xs1 + wi[1]); fv = sig(wf[0] * xs1 + wf[1]); gv = math.tanh(wg[0] * xs1 + wg[1])
ov = sig(wo[0] * xs1 + wo[1]); cv = fv * cp + iv * gv; hv = ov * math.tanh(cv)
Wt = np.array([[0.5], [1.0], [1.0], [0.5]]); Ut = np.zeros((4, 1))
hh, cc = lstm_cell(np.array([xs1]), np.array([hp]), np.array([cp]), Wt, Ut, np.zeros(4))
out.check("scalar LSTM step by hand", [hv, cv], [hh[0], cc[0]])
for k, v in dict(i=iv, f=fv, g=gv, o=ov, c=cv, h=hv).items():
    out.val(f"s_{k}", v, 4)

# ------------------------------------- gradient through time vs spectral radius
def grad_decay(rho, T=60, Hn=64, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    Wr = torch.randn(Hn, Hn, generator=g_)
    Wr = Wr * rho / torch.linalg.eigvals(Wr).abs().max()
    Wx = torch.randn(Hn, 8, generator=g_) * 0.3
    xseq = torch.randn(T, 8, generator=g_)
    hs_ = [torch.zeros(Hn, requires_grad=True)]
    for t in range(T):
        hs_.append(torch.tanh(Wr @ hs_[-1] + Wx @ xseq[t]))
        hs_[-1].retain_grad()
    hs_[-1].sum().backward()
    return np.array([hs_[t].grad.norm().item() for t in range(1, T + 1)])
cols = {"lag": np.arange(59, -1, -1)}
for rho in (0.5, 0.9, 1.5, 3.0):
    gd = grad_decay(rho)
    cols[f"r{str(rho).replace('.', '')}"] = np.log10(gd / gd[-1])
    out.val(f"gd_{str(rho).replace('.', '')}", sci(gd[0] / gd[-1], 1))
out.dat("graddecay", cols)
# LSTM cell-state path: dc_t/dc_{t-1} = f_t exactly (when gates are held fixed)
out.val("f095_50", 0.95 ** 50, 3); out.val("tanh_bound_50", sci(0.5 ** 50, 1))

# ------------------------------------------------------------ beam search
V, T = 3, 3
# a hand-made decoder: P(first token) and P(next | previous), tokens a, b, c
P0 = np.array([0.5, 0.4, 0.1])
Pn = np.array([[0.4, 0.3, 0.3],                  # after a
               [0.9, 0.05, 0.05],                # after b
               [0.34, 0.33, 0.33]])              # after c
def logp(prev, t):
    return np.log(P0 if t == 0 else Pn[prev])
# [[beam]]
def beam_search(k, T, start=0):
    beams = [((), 0.0)]                         # (tokens, total log-prob)
    for t in range(T):
        cand = []
        for seq, s in beams:
            prev = seq[-1] if seq else start
            lp = logp(prev, t)
            cand += [(seq + (v,), s + lp[v]) for v in range(V)]
        beams = sorted(cand, key=lambda c: -c[1])[:k]   # keep the k best prefixes
    return beams[0]
# [[/beam]]
greedy = beam_search(1, T)
beam2 = beam_search(2, T)
best = max(((seq, sum(logp(([0] + list(seq))[t], t)[seq[t]] for t in range(T)))
            for seq in itertools.product(range(V), repeat=T)), key=lambda c: c[1])
out.check("beam search with k = V^T equals brute force", beam_search(V ** T, T)[1], best[1])
out.val("greedy_seq", "".join("abc"[v] for v in greedy[0])); out.val("greedy_p", math.exp(greedy[1]), 4)
out.val("beam2_seq", "".join("abc"[v] for v in beam2[0])); out.val("beam2_p", math.exp(beam2[1]), 4)
out.val("best_seq", "".join("abc"[v] for v in best[0])); out.val("best_p", math.exp(best[1]), 4)
# find a seed where greedy is suboptimal is not needed: report whether it is
out.val("greedy_opt", "is" if abs(greedy[1] - best[1]) < 1e-12 else "is not")

# ---------------------------------------------------------------- CTC
# [[ctc]]
def collapse(path, blank=0):
    outp, prev = [], None
    for s in path:
        if s != prev and s != blank:            # merge repeats, then drop blanks
            outp.append(s)
        prev = s
    return tuple(outp)
# [[/ctc]]
Tc, Vc = 4, 3                                    # symbols: 0 = blank, 1 = 'a', 2 = 'b'
lg = rng.normal(size=(Tc, Vc))
lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
target = (1, 2)
paths = [p for p in itertools.product(range(Vc), repeat=Tc) if collapse(p) == target]
prob = sum(math.exp(sum(lp[t, s] for t, s in enumerate(p))) for p in paths)
ref = F.ctc_loss(torch.tensor(lp)[:, None, :], torch.tensor([target]), torch.tensor([Tc]),
                 torch.tensor([2]), reduction="sum").item()
out.check("CTC by enumerating alignments vs F.ctc_loss", -math.log(prob), ref)
out.val("ctc_npaths", len(paths)); out.val("ctc_total", Vc ** Tc); out.val("ctc_loss", -math.log(prob), 4)
out.tex("ctc_paths", ", ".join("".join("-ab"[s] for s in p) for p in paths))
out.val("collapse_demo", "".join("-helo"[s] for s in collapse((1, 1, 0, 2, 0, 3, 3, 0, 3, 4, 4))))

# ------------------------------------------------------------ problem values
blk = 64 * (50 + 64) + 2 * 64
out.check("problem: LSTM(50,64)", 4 * blk, n_params(nn.LSTM(50, 64)), atol=0)
out.check("problem: GRU(50,64)", 3 * blk, n_params(nn.GRU(50, 64)), atol=0)
out.val("pa_blk", thousands(blk)); out.val("pa_lstm", thousands(4 * blk)); out.val("pa_gru", thousands(3 * blk))
out.check("problem: Linear(128,5)", 128 * 5 + 5, n_params(nn.Linear(128, 5)), atol=0)
out.val("pb_lin", 128 * 5 + 5)
out.val("pf_beam", thousands(5 * 10000))
