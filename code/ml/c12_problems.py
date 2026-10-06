"""Numbers used in chapter 12's problems."""
import numpy as np
import torch
import torch.nn.functional as Fnn
from common import setup

out = setup(__file__)
for tau, key in [(1.0, "1"), (0.1, "01")]:
    logits = np.array([0.8, 0.5, 0.5, 0.5]) / tau
    loss = -(logits[0] - np.log(np.exp(logits).sum()))
    out.check(f"12C InfoNCE tau={tau} vs torch", loss, Fnn.cross_entropy(torch.tensor(logits)[None], torch.tensor([0])).item())
    out.val(f"p12c_{key}", loss, 4)
