"""Chapter 6b: the degradation problem. Plain vs residual MLPs (with BatchNorm) of
increasing depth on the same regression task: training loss and the gradient
norm reaching the first layer."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import setup

out = setup(__file__)
torch.set_default_dtype(torch.float32)

g = torch.Generator().manual_seed(0)
X = torch.rand(1024, 2, generator=g) * 4 - 2
y = (torch.sin(2 * X[:, :1]) * torch.cos(2 * X[:, 1:]) + 0.3 * X[:, :1] ** 2)


# [[resblock]]
class Block(nn.Module):
    def __init__(self, d, residual):
        super().__init__()
        self.lin, self.bn, self.residual = nn.Linear(d, d), nn.BatchNorm1d(d), residual

    def forward(self, h):
        f = torch.relu(self.bn(self.lin(h)))
        return h + f if self.residual else f      # the only difference: "+ h"
# [[/resblock]]


def make(depth, residual, width=32):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, width), *[Block(width, residual) for _ in range(depth)],
                         nn.Linear(width, 1))


def train(depth, residual, steps=300):
    net = make(depth, residual)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    gq = torch.Generator().manual_seed(1)
    losses = []
    for t in range(steps):
        idx = torch.randint(0, 1024, (128,), generator=gq)
        opt.zero_grad()
        loss = F.mse_loss(net(X[idx]), y[idx])
        loss.backward()
        if t == 0:
            g0 = net[0].weight.grad.norm().item()
        opt.step()
        if t % 5 == 4:
            net.eval()
            with torch.no_grad():
                losses.append(F.mse_loss(net(X), y).item())
            net.train()
    return np.array(losses), g0


depths = [4, 16, 64]
cols = {"step": np.arange(5, 301, 5)}
rows = []
for d in depths:
    for res in (False, True):
        L, g0 = train(d, res)
        tag = f"{'res' if res else 'plain'}{d}"
        out.val(f"final_{tag}", L[-5:].mean(), 4)
        out.val(f"g0_{tag}", g0, 2 if g0 < 1000 else 0)
        cols[tag] = L
out.dat("curves", cols)
