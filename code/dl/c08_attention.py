"""Chapter 8a: attention numerics. Scaled dot-product attention on 3 tokens by
hand (vs torch), the sqrt(d_k) argument by simulation, masks, multi-head
attention with shapes (vs nn.MultiheadAttention), permutation equivariance,
sinusoidal/RoPE/ALiBi, Transformer parameter counts (vs torch), complexity,
KV-cache memory, and blockwise attention with an online softmax."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import setup, thousands
from dlutil import human, sci

out = setup(__file__)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------------------------- 3-token attention by hand
# [[sdpa]]
def attention(Q, K, V, mask=None):
    """Q: (..., Tq, dk), K: (..., Tk, dk), V: (..., Tk, dv); mask True = blocked."""
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])     # (..., Tq, Tk) scores
    if mask is not None:
        S = np.where(mask, -np.inf, S)                         # block before the softmax
    A = softmax(S, axis=-1)                                    # each row sums to 1
    return A @ V, A
# [[/sdpa]]
Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Kx = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Vx = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
O, A = attention(Q, Kx, Vx)
Ot = F.scaled_dot_product_attention(torch.tensor(Q)[None], torch.tensor(Kx)[None], torch.tensor(Vx)[None])[0]
out.check("3-token attention vs F.scaled_dot_product_attention", O, Ot.numpy())
S = Q @ Kx.T / math.sqrt(2)
fm = lambda M, nd: " \\\\ ".join(" & ".join(f"{v:.{nd}f}" for v in row) for row in M)
out.tex("S", fm(S, 3)); out.tex("A", fm(A, 3)); out.tex("O", fm(O, 3))
out.tex("QKt", fm(Q @ Kx.T, 0))
out.val("o1_0", O[0, 0], 3); out.val("a1_0", A[0, 0], 3)
out.dat("heat3", {"x": np.repeat(np.arange(3), 3), "y": np.tile(np.arange(3), 3), "a": A.T.ravel()})
# causal mask
cm = np.triu(np.ones((3, 3), bool), 1)
Oc, Ac = attention(Q, Kx, Vx, cm)
Oct = F.scaled_dot_product_attention(torch.tensor(Q)[None], torch.tensor(Kx)[None], torch.tensor(Vx)[None], is_causal=True)[0]
out.check("causal attention vs is_causal=True", Oc, Oct.numpy())
out.tex("Ac", fm(Ac, 3)); out.tex("Oc", fm(Oc, 3))
# wrong: zero after the softmax
Abad = A * (~cm)
out.tex("Abad_rowsum", ", ".join(f"{v:.3f}" for v in Abad.sum(1)))
# padding mask: key 3 is padding
pm = np.zeros((3, 3), bool); pm[:, 2] = True
Op, Ap = attention(Q, Kx, Vx, pm)
Opt = F.scaled_dot_product_attention(torch.tensor(Q)[None], torch.tensor(Kx)[None], torch.tensor(Vx)[None],
                                     attn_mask=~torch.tensor(pm)[None])[0]
out.check("padding-masked attention vs attn_mask", Op, Opt.numpy())
out.tex("Ap", fm(Ap, 3))

# ----------------------------------------------------------- why sqrt(d_k)
ds = [4, 16, 64, 256, 1024]
var_dot, ent_raw, ent_scaled, max_raw, max_scaled = [], [], [], [], []
for dk in ds:
    q = rng.normal(size=(2000, dk)); k = rng.normal(size=(2000, dk))
    dots = np.sum(q * k, 1)
    var_dot.append(dots.var())
    qs = rng.normal(size=(400, dk)); ks = rng.normal(size=(400, 32, dk))
    s = np.einsum("nd,ntd->nt", qs, ks)
    for scale, ent, mx in ((1.0, ent_raw, max_raw), (math.sqrt(dk), ent_scaled, max_scaled)):
        a = softmax(s / scale)
        ent.append(np.mean(-np.sum(a * np.log(a + 1e-300), 1)))
        mx.append(np.mean(a.max(1)))
out.dat("sqrtdk", {"dk": ds, "var": var_dot, "ent_raw": ent_raw, "ent_scaled": ent_scaled,
                   "max_raw": max_raw, "max_scaled": max_scaled})
for dk, v, mr, ms in zip(ds, var_dot, max_raw, max_scaled):
    out.val(f"var_{dk}", v, 1); out.val(f"maxraw_{dk}", mr, 3); out.val(f"maxsc_{dk}", ms, 3)
out.val("ln32", math.log(32), 3)
out.val("ent_raw_1024", ent_raw[-1], 3); out.val("ent_sc_1024", ent_scaled[-1], 3)
slope = np.polyfit(np.log(ds), np.log(var_dot), 1)[0]
out.check("Var(q.k) grows linearly in d_k (log-log slope)", slope, 1.0, atol=0.05)
# gradient of softmax when saturated: max entry of Jacobian
def jac_norm(z):
    p = softmax(z); return np.linalg.norm(np.diag(p) - np.outer(p, p))
zz = rng.normal(size=32) * math.sqrt(512)
out.val("jac_raw", sci(jac_norm(zz), 1)); out.val("jac_scaled", jac_norm(zz / math.sqrt(512)), 4)

# ---------------------------------------------------- multi-head attention
d, h, T, B = 8, 2, 4, 1
dk = d // h
# [[mha]]
def mha(X, Wq, Wk, Wv, Wo, bq, bk, bv, bo, h, mask=None):
    """X: (B, T, d); W*: (d, d) applied as X @ W.T + b (nn.Linear layout)."""
    B, T, d = X.shape
    dk = d // h
    split = lambda Y: Y.reshape(B, T, h, dk).transpose(0, 2, 1, 3)   # (B, h, T, dk)
    Qh, Kh, Vh = (split(X @ W.T + b) for W, b in ((Wq, bq), (Wk, bk), (Wv, bv)))
    Oh, A = attention(Qh, Kh, Vh, mask)                              # (B, h, T, dk)
    concat = Oh.transpose(0, 2, 1, 3).reshape(B, T, d)               # (B, T, d)
    return concat @ Wo.T + bo, A
# [[/mha]]
ref = nn.MultiheadAttention(d, h, batch_first=True)
Wq, Wk, Wv = (w.detach().numpy() for w in ref.in_proj_weight.chunk(3))
bq, bk, bv = (b_.detach().numpy() for b_ in ref.in_proj_bias.chunk(3))
Wo, bo = ref.out_proj.weight.detach().numpy(), ref.out_proj.bias.detach().numpy()
X = rng.normal(size=(B, T, d))
ours, Aours = mha(X, Wq, Wk, Wv, Wo, bq, bk, bv, bo, h)
Xt = torch.tensor(X)
rt, at = ref(Xt, Xt, Xt, average_attn_weights=False)
out.check("NumPy MHA vs nn.MultiheadAttention (output)", ours, rt.detach().numpy())
out.check("NumPy MHA vs nn.MultiheadAttention (weights)", Aours, at.detach().numpy())
cmask = np.triu(np.ones((T, T), bool), 1)
oc, _ = mha(X, Wq, Wk, Wv, Wo, bq, bk, bv, bo, h, cmask)
rc, _ = ref(Xt, Xt, Xt, attn_mask=torch.tensor(cmask))
out.check("NumPy causal MHA vs nn.MultiheadAttention(attn_mask)", oc, rc.detach().numpy())
out.val("mha_params", sum(p.numel() for p in ref.parameters()))
out.val("mha_formula", 4 * d * d + 4 * d)
# permutation equivariance without positional information
perm = np.array([2, 0, 3, 1])
op, _ = mha(X[:, perm], Wq, Wk, Wv, Wo, bq, bk, bv, bo, h)
out.check("self-attention is permutation-equivariant", op, ours[:, perm])
for dm, hh in ((512, 8), (768, 12), (1024, 16)):
    m = nn.MultiheadAttention(dm, hh)
    out.check(f"MHA d={dm} params", sum(p.numel() for p in m.parameters()), 4 * dm * dm + 4 * dm, atol=0)
    out.val(f"mha_{dm}", thousands(4 * dm * dm + 4 * dm))

# ------------------------------------------------------- positional encodings
# [[sinusoid]]
def sinusoidal(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / 10000 ** (2 * i / d)
    PE = np.zeros((T, d))
    PE[:, 0::2], PE[:, 1::2] = np.sin(angle), np.cos(angle)
    return PE
# [[/sinusoid]]
PE = sinusoidal(50, 32)
out.dat("pe", {"pos": np.repeat(np.arange(50), 32), "dim": np.tile(np.arange(32), 50), "v": PE.ravel()})
pe8 = sinusoidal(3, 8)
out.tex("pe_row1", ", ".join(f"{v:.4f}" for v in pe8[1]))
out.tex("pe_row2", ", ".join(f"{v:.4f}" for v in pe8[2]))
# relative-position property of sinusoids: PE(p+k) is a fixed rotation of PE(p)
k_off = 5
Rm = np.zeros((32, 32))
for j in range(16):
    w = 1 / 10000 ** (2 * j / 32)
    c, s_ = math.cos(w * k_off), math.sin(w * k_off)
    Rm[2 * j:2 * j + 2, 2 * j:2 * j + 2] = [[c, s_], [-s_, c]]
out.check("PE(p+k) = R_k PE(p) for all p", PE[k_off:], (Rm @ PE[:-k_off].T).T)
# RoPE
# [[rope]]
def rope(x, pos, base=10000.0):
    """Rotate consecutive pairs (x0,x1), (x2,x3), ... by angle pos * theta_j."""
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(pos * theta), np.sin(pos * theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = x1 * c - x2 * s, x1 * s + x2 * c
    return out
# [[/rope]]
qv, kv = rng.normal(size=16), rng.normal(size=16)
dots = [rope(qv, m) @ rope(kv, n) for m, n in ((3, 1), (10, 8), (52, 50))]
out.check("RoPE: q_m . k_n depends only on m - n", dots, [dots[0]] * 3)
out.check("RoPE preserves norms", np.linalg.norm(rope(qv, 7)), np.linalg.norm(qv))
out.val("rope_dot", dots[0], 4)
# ALiBi slopes for 8 heads
slopes = [2 ** (-8 * (i + 1) / 8) for i in range(8)]
out.val("alibi_slopes", ", ".join(f"{s_:g}" for s_ in slopes[:4]) + ", \\dots, " + f"{slopes[-1]:g}")

# -------------------------------------------------- Transformer parameter counts
# [[count]]
def enc_layer_params(d, dff):
    attn = 4 * d * d + 4 * d                 # W_Q, W_K, W_V, W_O and their biases
    ffn = d * dff + dff + dff * d + d        # two linear layers
    ln = 2 * (2 * d)                         # two LayerNorms (gamma, beta)
    return attn + ffn + ln

def dec_layer_params(d, dff):
    return enc_layer_params(d, dff) + (4 * d * d + 4 * d) + 2 * d   # + cross-attn + 3rd LN
# [[/count]]
for d_, dff_ in ((512, 2048), (768, 3072)):
    e = nn.TransformerEncoderLayer(d_, 8, dff_)
    dl = nn.TransformerDecoderLayer(d_, 8, dff_)
    out.check(f"encoder layer d={d_}", enc_layer_params(d_, dff_), sum(p.numel() for p in e.parameters()), atol=0)
    out.check(f"decoder layer d={d_}", dec_layer_params(d_, dff_), sum(p.numel() for p in dl.parameters()), atol=0)
out.val("enc512", thousands(enc_layer_params(512, 2048))); out.val("dec512", thousands(dec_layer_params(512, 2048)))
out.val("attn512", thousands(4 * 512 * 512 + 4 * 512)); out.val("ffn512", thousands(2 * 512 * 2048 + 2048 + 512))
tf = nn.Transformer(512, 8, 6, 6, 2048, batch_first=True)
n_tf = sum(p.numel() for p in tf.parameters())
ours_tf = 6 * enc_layer_params(512, 2048) + 6 * dec_layer_params(512, 2048) + 2 * (2 * 512)   # + final LNs
out.check("nn.Transformer(512, 8, 6+6, 2048) body", ours_tf, n_tf, atol=0)
out.val("tf_body", human(n_tf)); out.val("tf_body_exact", thousands(n_tf))
V = 37000
out.val("tf_emb", human(V * 512)); out.val("tf_total_shared", human(n_tf + V * 512))
# a BERT-base-shaped encoder: L=12, d=768, ff=3072, V=30522, 512 positions, 2 segments, pooler
L_, d_, ff_ = 12, 768, 3072
emb = 30522 * d_ + 512 * d_ + 2 * d_ + 2 * d_          # token, position, segment, embedding LN
body = L_ * enc_layer_params(d_, ff_)
pool = d_ * d_ + d_
enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_, 12, ff_), L_, enable_nested_tensor=False)
out.check("12-layer d=768 encoder stack", body, sum(p.numel() for p in enc.parameters()), atol=0)
out.val("bert_emb", human(emb)); out.val("bert_body", human(body)); out.val("bert_total", human(emb + body + pool))
out.val("bert_layer", thousands(enc_layer_params(d_, ff_)))
# a GPT-2-small-shaped decoder-only model: L=12, d=768, V=50257, ctx 1024, tied output
emb_g = 50257 * d_ + 1024 * d_
out.val("gpt_total", human(emb_g + L_ * enc_layer_params(d_, ff_) + 2 * d_))
out.val("gpt_emb_frac", 100 * emb_g / (emb_g + L_ * enc_layer_params(d_, ff_) + 2 * d_), 0)
out.val("twelve_d2", human(12 * d_ * d_))

# ---------------------------------------------------------- complexity and KV cache
for n_ in (512, 4096, 32768):
    out.val(f"attn_flops_{n_}", human(4 * n_ * n_ * 768))
    out.val(f"proj_flops_{n_}", human(8 * n_ * 768 * 768))
    out.val(f"attnmat_{n_}", human(n_ * n_ * 12 * 2, 1).replace("\\,\\mathrm", "\\,\\mathrm"))
# KV cache for a config: 32 layers, 32 heads of 128, fp16, 4096 tokens, batch 1
Lk, Hk, dh, Tk, bytes_ = 32, 32, 128, 4096, 2
kv = 2 * Lk * Hk * dh * Tk * bytes_
out.val("kv_bytes", f"{kv / 2**30:.0f}"); out.val("kv_per_tok", f"{2 * Lk * Hk * dh * bytes_ / 2**20:.1f}")
out.val("kv_gqa8", f"{2 * Lk * 8 * dh * Tk * bytes_ / 2**30:.1f}")
out.val("kv_mqa", f"{2 * Lk * 1 * dh * Tk * bytes_ / 2**30:.3f}")

# ---------------------------------------------- blockwise attention, online softmax
# [[online]]
def blockwise_attention(Q, K, V, block=4):
    """Process keys/values in blocks, keeping a running max m, running
    normaliser l and running output O for each query (the FlashAttention idea)."""
    Tq, dk = Q.shape
    m = np.full(Tq, -np.inf); l = np.zeros(Tq); O = np.zeros((Tq, V.shape[1]))
    for s in range(0, K.shape[0], block):
        Sb = Q @ K[s:s + block].T / np.sqrt(dk)              # scores for this block only
        m_new = np.maximum(m, Sb.max(1))
        scale = np.exp(m - m_new)                            # rescale old partial sums
        P = np.exp(Sb - m_new[:, None])
        l = l * scale + P.sum(1)
        O = O * scale[:, None] + P @ V[s:s + block]
        m = m_new
    return O / l[:, None]
# [[/online]]
Qb, Kb, Vb = rng.normal(size=(6, 8)), rng.normal(size=(17, 8)), rng.normal(size=(17, 5))
out.check("blockwise online-softmax attention == standard", blockwise_attention(Qb, Kb, Vb, 4), attention(Qb, Kb, Vb)[0])

# ------------------------------------------------------------ problem values
Qp = np.array([[1.0, 0.0], [0.0, 2.0]]); Kp = np.array([[1.0, 1.0], [0.0, 1.0]]); Vp = np.array([[10.0], [20.0]])
Op_, Ap_ = attention(Qp, Kp, Vp)
out.check("problem A vs torch", Op_, F.scaled_dot_product_attention(torch.tensor(Qp)[None], torch.tensor(Kp)[None],
                                                                     torch.tensor(Vp)[None])[0].numpy())
out.val("pa_w1", ", ".join(f"{v:.3f}" for v in Ap_[0])); out.val("pa_o1", Op_[0, 0], 2)
out.val("pa_w2", ", ".join(f"{v:.3f}" for v in Ap_[1])); out.val("pa_o2", Op_[1, 0], 2)
mb = nn.MultiheadAttention(256, 4)
out.check("problem B MHA(256)", sum(p.numel() for p in mb.parameters()), 4 * 256 ** 2 + 4 * 256, atol=0)
out.val("pb_attn", thousands(4 * 256 ** 2 + 4 * 256)); out.val("pb_ffn", thousands(2 * 256 * 1024 + 1024 + 256))
sc = np.array([8.0, 4.0, 0.0, -4.0])
out.val("pc_raw", ", ".join(f"{v:.4f}" for v in softmax(sc)))
out.val("pc_scaled", ", ".join(f"{v:.3f}" for v in softmax(sc / 8)))
kvp = 2 * 24 * 16 * 64 * 2048 * 8 * 2
out.val("pf_kv", f"{kvp / 2**30:.2f}"); out.val("pf_kv4", f"{kvp / 4 / 2**30:.3f}")
