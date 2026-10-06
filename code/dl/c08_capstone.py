"""Chapter 8 capstone: an encoder-decoder Transformer written twice, in PyTorch
(trained) and in NumPy (forward only, using the trained weights), on a toy task:
reverse a digit sequence of length 3-8. Checks: our MHA vs nn.MultiheadAttention,
our layers vs nn.TransformerEncoderLayer / DecoderLayer, NumPy forward vs PyTorch
forward, NumPy greedy decoding vs PyTorch greedy decoding."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import setup, thousands

out = setup(__file__)
torch.set_default_dtype(torch.float32)
PAD, BOS, EOS, NDIG = 0, 1, 2, 10
VOCAB = 3 + NDIG
LMIN, LMAX = 3, 8


def make_batch(B, g):
    lens = torch.randint(LMIN, LMAX + 1, (B, 1), generator=g)
    digits = torch.randint(3, 3 + NDIG, (B, LMAX), generator=g)
    pos = torch.arange(LMAX)[None]
    valid = pos < lens
    src = torch.where(valid, digits, PAD)
    rev = torch.gather(digits, 1, (lens - 1 - pos).clamp(min=0))     # reversed digits
    tgt = torch.full((B, LMAX + 2), PAD)
    tgt[:, 0] = BOS
    tgt[:, 1:LMAX + 1] = torch.where(valid, rev, PAD)
    tgt.scatter_(1, lens + 1, EOS)
    return src, tgt


# ======================================================= PyTorch implementation
# [[torch_model]]
class MHA(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, mem, mask):              # mask: True = blocked, (B|1, 1, Tq, Tk)
        B, Tq, d = x.shape
        Tk = mem.shape[1]
        split = lambda y, T: y.view(B, T, self.h, self.dk).transpose(1, 2)   # (B, h, T, dk)
        Q, K, V = split(self.q(x), Tq), split(self.k(mem), Tk), split(self.v(mem), Tk)
        S = Q @ K.transpose(-1, -2) / math.sqrt(self.dk)                  # (B, h, Tq, Tk)
        A = torch.softmax(S.masked_fill(mask, float("-inf")), dim=-1)
        self.last_attn = A.detach()
        return self.o((A @ V).transpose(1, 2).reshape(B, Tq, d))

class FFN(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d, dff), nn.Linear(dff, d)
    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))

class EncoderLayer(nn.Module):                     # pre-LN: x + f(LN(x))
    def __init__(self, d, h, dff):
        super().__init__()
        self.ln1, self.att, self.ln2, self.ffn = nn.LayerNorm(d), MHA(d, h), nn.LayerNorm(d), FFN(d, dff)
    def forward(self, x, src_mask):
        y = self.ln1(x)
        x = x + self.att(y, y, src_mask)
        return x + self.ffn(self.ln2(x))

class DecoderLayer(nn.Module):
    def __init__(self, d, h, dff):
        super().__init__()
        self.ln1, self.self_att = nn.LayerNorm(d), MHA(d, h)
        self.ln2, self.cross_att = nn.LayerNorm(d), MHA(d, h)
        self.ln3, self.ffn = nn.LayerNorm(d), FFN(d, dff)
    def forward(self, y, mem, tgt_mask, src_mask):
        z = self.ln1(y)
        y = y + self.self_att(z, z, tgt_mask)              # masked self-attention
        y = y + self.cross_att(self.ln2(y), mem, src_mask)  # queries from decoder, keys/values from encoder
        return y + self.ffn(self.ln3(y))

def sinusoidal(T, d):
    pos = torch.arange(T)[:, None].float(); i = torch.arange(d // 2)[None].float()
    ang = pos / 10000 ** (2 * i / d)
    pe = torch.zeros(T, d); pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe

class Transformer(nn.Module):
    def __init__(self, V, d=64, h=4, dff=128, n_enc=2, n_dec=2, Tmax=16):
        super().__init__()
        self.d, self.emb = d, nn.Embedding(V, d)            # shared by encoder, decoder, output
        nn.init.normal_(self.emb.weight, std=d ** -0.5)     # so that emb * sqrt(d) has unit scale
        self.register_buffer("pe", sinusoidal(Tmax, d))
        self.enc = nn.ModuleList(EncoderLayer(d, h, dff) for _ in range(n_enc))
        self.dec = nn.ModuleList(DecoderLayer(d, h, dff) for _ in range(n_dec))
        self.ln_enc, self.ln_dec = nn.LayerNorm(d), nn.LayerNorm(d)

    def embed(self, tok):
        return self.emb(tok) * math.sqrt(self.d) + self.pe[:tok.shape[1]]

    def encode(self, src):
        src_mask = (src == PAD)[:, None, None, :]           # (B, 1, 1, Ts): hide padding keys
        x = self.embed(src)
        for layer in self.enc:
            x = layer(x, src_mask)
        return self.ln_enc(x), src_mask

    def decode(self, tgt_in, mem, src_mask):
        T = tgt_in.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)[None, None]
        tgt_mask = causal | (tgt_in == PAD)[:, None, None, :]
        y = self.embed(tgt_in)
        for layer in self.dec:
            y = layer(y, mem, tgt_mask, src_mask)
        return self.ln_dec(y) @ self.emb.weight.T          # tied output projection -> logits

    def forward(self, src, tgt_in):
        mem, src_mask = self.encode(src)
        return self.decode(tgt_in, mem, src_mask)
# [[/torch_model]]


# --------------------------------------- our layers vs PyTorch's reference layers
torch.manual_seed(0)
d, h, dff = 64, 4, 128
ours = MHA(d, h); ref = nn.MultiheadAttention(d, h, batch_first=True)
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([ours.q.weight, ours.k.weight, ours.v.weight]))
    ref.in_proj_bias.copy_(torch.cat([ours.q.bias, ours.k.bias, ours.v.bias]))
    ref.out_proj.weight.copy_(ours.o.weight); ref.out_proj.bias.copy_(ours.o.bias)
x = torch.randn(3, 7, d); m = torch.randn(3, 5, d)
kpm = torch.zeros(3, 5, dtype=torch.bool); kpm[0, 3:] = True
o1 = ours(x, m, kpm[:, None, None, :])
o2, _ = ref(x, m, m, key_padding_mask=kpm)
out.check("our MHA (cross, key padding) vs nn.MultiheadAttention", o1.detach().numpy(), o2.detach().numpy(), atol=1e-5)

def copy_ln(dst, src_):
    dst.weight.copy_(src_.weight); dst.bias.copy_(src_.bias)
def copy_mha(ref_mha, our):
    ref_mha.in_proj_weight.copy_(torch.cat([our.q.weight, our.k.weight, our.v.weight]))
    ref_mha.in_proj_bias.copy_(torch.cat([our.q.bias, our.k.bias, our.v.bias]))
    ref_mha.out_proj.weight.copy_(our.o.weight); ref_mha.out_proj.bias.copy_(our.o.bias)
el = EncoderLayer(d, h, dff)
rel = nn.TransformerEncoderLayer(d, h, dff, dropout=0.0, batch_first=True, norm_first=True)
with torch.no_grad():
    copy_mha(rel.self_attn, el.att); copy_ln(rel.norm1, el.ln1); copy_ln(rel.norm2, el.ln2)
    rel.linear1.weight.copy_(el.ffn.l1.weight); rel.linear1.bias.copy_(el.ffn.l1.bias)
    rel.linear2.weight.copy_(el.ffn.l2.weight); rel.linear2.bias.copy_(el.ffn.l2.bias)
rel.eval()
spm = torch.zeros(3, 7, dtype=torch.bool); spm[1, 5:] = True
out.check("our pre-LN encoder layer vs nn.TransformerEncoderLayer(norm_first=True)",
          el(x, spm[:, None, None, :]).detach().numpy(), rel(x, src_key_padding_mask=spm).detach().numpy(), atol=1e-5)
dlr = DecoderLayer(d, h, dff)
rdl = nn.TransformerDecoderLayer(d, h, dff, dropout=0.0, batch_first=True, norm_first=True)
with torch.no_grad():
    copy_mha(rdl.self_attn, dlr.self_att); copy_mha(rdl.multihead_attn, dlr.cross_att)
    copy_ln(rdl.norm1, dlr.ln1); copy_ln(rdl.norm2, dlr.ln2); copy_ln(rdl.norm3, dlr.ln3)
    rdl.linear1.weight.copy_(dlr.ffn.l1.weight); rdl.linear1.bias.copy_(dlr.ffn.l1.bias)
    rdl.linear2.weight.copy_(dlr.ffn.l2.weight); rdl.linear2.bias.copy_(dlr.ffn.l2.bias)
rdl.eval()
causal = torch.triu(torch.ones(7, 7, dtype=torch.bool), 1)
out.check("our pre-LN decoder layer vs nn.TransformerDecoderLayer(norm_first=True)",
          dlr(x, m, causal[None, None], kpm[:, None, None, :]).detach().numpy(),
          rdl(x, m, tgt_mask=causal, memory_key_padding_mask=kpm).detach().numpy(), atol=1e-5)

# ===================================================================== training
torch.manual_seed(0)
model = Transformer(VOCAB)
n_params = sum(p.numel() for p in model.parameters())
out.val("n_params", thousands(n_params))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01, foreach=True)
STEPS, WARM = 600, 60
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda t: min((t + 1) / WARM, 0.5 * (1 + math.cos(math.pi * min(t, STEPS) / STEPS))))
g = torch.Generator().manual_seed(1)
losses = []
# [[train]]
for step in range(STEPS):
    src, tgt = make_batch(64, g)
    logits = model(src, tgt[:, :-1])                       # teacher forcing: input is shifted target
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt[:, 1:].reshape(-1), ignore_index=PAD)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    losses.append(loss.item())
# [[/train]]
out.dat("loss", {"step": np.arange(1, STEPS + 1)[::10], "loss": np.convolve(losses, np.ones(10) / 10, "same")[::10]})
out.val("loss_first", losses[0], 3); out.val("loss_last", np.mean(losses[-50:]), 4)
out.val("ln_vocab", math.log(VOCAB), 3)
out.val("steps", STEPS)


# [[greedy]]
@torch.no_grad()
def greedy_torch(model, src):
    mem, src_mask = model.encode(src)
    ys = torch.full((src.shape[0], 1), BOS)
    for _ in range(LMAX + 1):
        nxt = model.decode(ys, mem, src_mask)[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], 1)
    return ys
# [[/greedy]]

def exact_match(pred, tgt):
    ok = []
    for p, t in zip(pred.tolist(), tgt.tolist()):
        t = t[1:t.index(EOS) + 1]
        ok.append(p[1:len(t) + 1] == t)
    return np.array(ok)

model.eval()
gt = torch.Generator().manual_seed(123)
src_te, tgt_te = make_batch(1000, gt)
pred = greedy_torch(model, src_te)
em = exact_match(pred, tgt_te)
out.val("acc_test", 100 * em.mean(), 1)
tok_acc = (pred[:, 1:LMAX + 2] == tgt_te[:, 1:])[tgt_te[:, 1:] != PAD].float().mean().item()
out.val("tok_acc", 100 * tok_acc, 2)
ex_src = src_te[0].tolist(); ex_pred = pred[0].tolist()
fmt_seq = lambda s: " ".join(str(t - 3) for t in s if t >= 3)
out.val("ex_src", fmt_seq(ex_src)); out.val("ex_pred", fmt_seq(ex_pred[1:ex_pred.index(EOS)] if EOS in ex_pred else ex_pred[1:]))

# cross-attention heat-map (last decoder layer, averaged over heads) for an 8-digit input
i8 = next(i for i in range(1000) if (src_te[i] != PAD).sum() == LMAX)
model.decode(tgt_te[i8:i8 + 1, :-1], *model.encode(src_te[i8:i8 + 1]))
A = model.dec[-1].cross_att.last_attn[0].mean(0).numpy()            # (Tt, Ts)
Tt = LMAX + 1
out.dat("xattn", {"src": np.tile(np.arange(LMAX), Tt), "tgt": np.repeat(np.arange(Tt), LMAX), "a": A[:Tt, :LMAX].ravel()})
anti = np.mean([A[t, LMAX - 1 - t] for t in range(LMAX)])
out.val("anti_diag", anti, 3)
out.val("x_src", fmt_seq(src_te[i8].tolist()))


# ===================================================================== NumPy port
# [[numpy_model]]
def np_layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b

def np_softmax(s):
    s = s - s.max(-1, keepdims=True); e = np.exp(s)
    return e / e.sum(-1, keepdims=True)

def np_mha(P, pre, x, mem, mask, h):
    B, Tq, d = x.shape; Tk = mem.shape[1]; dk = d // h
    lin = lambda z, n: z @ P[f"{pre}.{n}.weight"].T + P[f"{pre}.{n}.bias"]
    split = lambda z, T: z.reshape(B, T, h, dk).transpose(0, 2, 1, 3)
    Q, K, V = split(lin(x, "q"), Tq), split(lin(mem, "k"), Tk), split(lin(mem, "v"), Tk)
    S = np.where(mask, -np.inf, Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dk))
    O = (np_softmax(S) @ V).transpose(0, 2, 1, 3).reshape(B, Tq, d)
    return lin(O, "o")

def np_ffn(P, pre, x):
    hdn = np.maximum(0, x @ P[f"{pre}.l1.weight"].T + P[f"{pre}.l1.bias"])
    return hdn @ P[f"{pre}.l2.weight"].T + P[f"{pre}.l2.bias"]

def np_forward(P, src, tgt_in, h=4, n_enc=2, n_dec=2):
    d = P["emb.weight"].shape[1]
    ln = lambda z, n: np_layernorm(z, P[f"{n}.weight"], P[f"{n}.bias"])
    embed = lambda t: P["emb.weight"][t] * np.sqrt(d) + P["pe"][:t.shape[1]]
    src_mask = (src == PAD)[:, None, None, :]
    x = embed(src)
    for l in range(n_enc):
        y = ln(x, f"enc.{l}.ln1"); x = x + np_mha(P, f"enc.{l}.att", y, y, src_mask, h)
        x = x + np_ffn(P, f"enc.{l}.ffn", ln(x, f"enc.{l}.ln2"))
    mem = ln(x, "ln_enc")
    T = tgt_in.shape[1]
    tgt_mask = np.triu(np.ones((T, T), bool), 1)[None, None] | (tgt_in == PAD)[:, None, None, :]
    y = embed(tgt_in)
    for l in range(n_dec):
        z = ln(y, f"dec.{l}.ln1"); y = y + np_mha(P, f"dec.{l}.self_att", z, z, tgt_mask, h)
        y = y + np_mha(P, f"dec.{l}.cross_att", ln(y, f"dec.{l}.ln2"), mem, src_mask, h)
        y = y + np_ffn(P, f"dec.{l}.ffn", ln(y, f"dec.{l}.ln3"))
    return ln(y, "ln_dec") @ P["emb.weight"].T
# [[/numpy_model]]

model64 = model.double()
P = {k: v.detach().numpy() for k, v in model64.state_dict().items()}
src_c, tgt_c = src_te[:50], tgt_te[:50]
with torch.no_grad():
    lt = model64(src_c, tgt_c[:, :-1]).numpy()
ln_ = np_forward(P, src_c.numpy(), tgt_c[:, :-1].numpy())
out.check("NumPy Transformer forward vs PyTorch (trained weights, 50 sequences)", ln_, lt, atol=1e-9)

def greedy_np(P, src):
    ys = np.full((src.shape[0], 1), BOS)
    for _ in range(LMAX + 1):
        nxt = np_forward(P, src, ys)[:, -1].argmax(-1)[:, None]
        ys = np.concatenate([ys, nxt], 1)
    return ys
pn = greedy_np(P, src_te[:200].numpy())
pt = greedy_torch(model64, src_te[:200]).numpy()
out.check("NumPy greedy decoding == PyTorch greedy decoding (200 sequences)", pn, pt, atol=0)
out.val("np_acc200", 100 * exact_match(torch.tensor(pn), tgt_te[:200]).mean(), 1)
