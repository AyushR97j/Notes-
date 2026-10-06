"""Chapter 7b: long-range memory at initialisation. How much does the input at
step 1 influence the final hidden state, relative to the input at step T, for a
vanilla RNN, a default LSTM and an LSTM whose forget-gate bias is set high?"""
import numpy as np
import torch
import torch.nn as nn
from common import setup
from dlutil import sci

out = setup(__file__)
torch.set_default_dtype(torch.float64)
H, B = 32, 256


def make(kind, fbias=None):
    torch.manual_seed(0)
    m = (nn.LSTM if kind == "lstm" else nn.RNN)(1, H, batch_first=True)
    if fbias is not None:
        with torch.no_grad():                     # PyTorch gate order: i, f, g, o
            m.bias_ih_l0[H:2 * H].fill_(fbias)
            m.bias_hh_l0[H:2 * H].zero_()
    return m


def influence(m, T):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, T, 1, generator=g, requires_grad=True)
    o, _ = m(x)
    readout = torch.randn(H, generator=g)
    (o[:, -1] @ readout).sum().backward()
    gx = x.grad.abs().mean(0)[:, 0]
    return (gx[0] / gx[-1]).item()


Ts = [2, 5, 10, 20, 40, 60, 80, 100]
models = {"rnn": make("rnn"), "lstm": make("lstm"), "lstmf3": make("lstm", 3.0)}
cols = {"T": Ts}
for k, m in models.items():
    r = np.array([influence(m, T) for T in Ts])
    cols[k] = np.log10(r)
    out.val(f"{k}_20", sci(r[3], 1) if r[3] < 0.01 else f"{r[3]:.2f}")
    out.val(f"{k}_100_log", np.log10(r[-1]), 1)
out.dat("memory", cols)
out.val("sig3", 1 / (1 + np.exp(-3.0)), 3)
out.val("sig0", 0.5, 1)
