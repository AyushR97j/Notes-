"""Chapter 9: language models. BPE from scratch, perplexity, MLM masking counts,
decoding (temperature, top-k, top-p), LoRA counts and merge check, int8
quantisation vs torch, distillation loss vs F.kl_div, MoE active parameters,
6ND compute, BLEU and ROUGE-L by hand, a tiny TF-IDF retriever."""
import math
import warnings
warnings.filterwarnings("ignore", message=".*quantize.*")
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import gmean
from sklearn.feature_extraction.text import TfidfVectorizer
from common import setup, thousands
from dlutil import human, sci

out = setup(__file__)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------- BPE
corpus = {"deep": 6, "deeper": 3, "deepest": 2, "keep": 4, "keeper": 2, "sleep": 3}
# [[bpe]]
def bpe_train(word_freq, n_merges):
    words = {tuple(w) + ("</w>",): f for w, f in word_freq.items()}   # start from characters
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for sym, f in words.items():
            for a, b in zip(sym, sym[1:]):
                pairs[a, b] += f                       # count adjacent pairs, weighted by frequency
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))  # most frequent pair (ties: lexicographic)
        merges.append((best, pairs[best]))
        new = {}
        for sym, f in words.items():                   # replace the pair everywhere
            out_, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    out_.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out_.append(sym[i]); i += 1
            new[tuple(out_)] = f
        words = new
    return merges, words

def bpe_encode(word, merges):
    sym = list(word) + ["</w>"]
    for (a, b), _ in merges:                           # apply merges in learned order
        i = 0
        while i < len(sym) - 1:
            if sym[i] == a and sym[i + 1] == b:
                sym[i:i + 2] = [a + b]
            else:
                i += 1
    return sym
# [[/bpe]]
merges, words = bpe_train(corpus, 8)
fmtm = lambda p: "\\texttt{" + (p[0] + p[1]).replace("</w>", "\\_") + "}"
out.tex("bpe_merges", ", ".join(f"{fmtm(p)} ({c})" for p, c in merges))
total_before = sum(f * (len(w) + 1) for w, f in corpus.items())
total_after = sum(f * len(s) for s, f in words.items())
out.val("bpe_before", total_before); out.val("bpe_after", total_after)
for w in ("sleeper", "keepest", "deepen"):
    enc = bpe_encode(w, merges)
    out.check(f"BPE round trip '{w}'", float("".join(enc).replace("</w>", "") == w), 1.0)
    out.val(f"enc_{w}", " ".join(t.replace("</w>", "\\_") for t in enc))
# encoding the training words reproduces the training segmentation
for w, f in corpus.items():
    seg = tuple(bpe_encode(w, merges))
    assert seg in words, (w, seg)
out.check("BPE encode == training segmentation for every corpus word", 1.0, 1.0)

# --------------------------------------------------------------- perplexity
p_tok = np.array([0.5, 0.25, 0.1, 0.4, 0.8])
nll = -np.log(p_tok)
ppl = math.exp(nll.mean())
out.check("perplexity = exp(mean NLL) = 1/geometric-mean(p)", ppl, 1 / gmean(p_tok))
out.val("ppl", ppl, 3); out.val("ppl_ce", nll.mean(), 4)
out.val("ppl_uniform", 50000)
logits = torch.zeros(1, 7, 50000); tgt = torch.randint(0, 50000, (1, 7))
out.check("uniform model: PPL = V", math.exp(F.cross_entropy(logits.view(-1, 50000), tgt.view(-1)).item()), 50000.0, rtol=1e-9)

# -------------------------------------------------------- BERT-style masking
N = 100000
sel = rng.random(N) < 0.15
r = rng.random(N)
n_mask = int(np.sum(sel & (r < 0.8))); n_rand = int(np.sum(sel & (r >= 0.8) & (r < 0.9)))
n_keep = int(np.sum(sel & (r >= 0.9)))
out.val("mlm_sel", 100 * sel.mean(), 1); out.val("mlm_mask", 100 * n_mask / N, 1)
out.val("mlm_rand", 100 * n_rand / N, 1); out.val("mlm_keep", 100 * n_keep / N, 1)

# ---------------------------------------------------------------- decoding
z = np.array([3.0, 2.5, 1.0, 0.5, 0.0, -1.0])
def softmax(v):
    e = np.exp(v - v.max()); return e / e.sum()
# [[sampling]]
def temperature(z, T):
    return softmax(z / T)

def top_k(z, k):
    keep = np.argsort(-z)[:k]
    zz = np.full_like(z, -np.inf); zz[keep] = z[keep]
    return softmax(zz)

def top_p(z, p):                                   # nucleus sampling
    probs = softmax(z)
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    n = int(np.searchsorted(cum, p) + 1)           # smallest prefix with mass >= p
    zz = np.full_like(z, -np.inf); zz[order[:n]] = z[order[:n]]
    return softmax(zz)
# [[/sampling]]
def top_p_ref(z, p):                               # brute-force reference: try all prefix sizes
    probs = softmax(z); order = np.argsort(-probs)
    for n in range(1, len(z) + 1):
        if probs[order[:n]].sum() >= p - 1e-12:
            q = np.zeros_like(probs); q[order[:n]] = probs[order[:n]]
            return q / q.sum()
out.check("top-p vs brute-force reference", top_p(z, 0.9), top_p_ref(z, 0.9))
out.check("top-k(k=V) == softmax", top_k(z, 6), softmax(z))
out.check("temperature 1 == softmax (torch)", temperature(z, 1.0), torch.softmax(torch.tensor(z), 0).numpy())
fv = lambda v: ", ".join(f"{x:.3f}" for x in v)
out.val("dec_p", fv(softmax(z))); out.val("dec_t05", fv(temperature(z, 0.5))); out.val("dec_t2", fv(temperature(z, 2.0)))
out.val("dec_k2", fv(top_k(z, 2))); out.val("dec_p09", fv(top_p(z, 0.9)))
out.val("dec_p09_n", int(np.sum(top_p(z, 0.9) > 0)))
out.val("dec_cum", fv(np.cumsum(np.sort(softmax(z))[::-1])))

# ------------------------------------------------------------------- LoRA
# [[lora]]
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # frozen pretrained weight
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero: starts as a no-op
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
# [[/lora]]
torch.manual_seed(0)
base = nn.Linear(64, 32)
lora = LoRALinear(base, r=4, alpha=8)
xx = torch.randn(5, 64)
out.check("LoRA with B = 0 equals the base layer", lora(xx).detach().numpy(), base(xx).detach().numpy())
with torch.no_grad():
    lora.B.normal_()
merged = base.weight + lora.scale * lora.B @ lora.A
out.check("merged weight W + (a/r) B A gives the same output", lora(xx).detach().numpy(),
          (xx @ merged.T + base.bias).detach().numpy())
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
out.check("LoRA trainable params = r (d_in + d_out)", trainable, 4 * (64 + 32), atol=0)
d = 4096
for r_ in (4, 8, 16, 64):
    out.val(f"lora_{r_}", thousands(r_ * 2 * d)); out.val(f"lora_pct_{r_}", 100 * r_ * 2 * d / d ** 2, 2)
out.val("lora_full", thousands(d * d))
# a whole model: 32 layers, LoRA r=8 on W_Q and W_V (d=4096)
L = 32
full = L * 12 * d * d
lo = L * 2 * (8 * 2 * d)
out.val("lora_model_full", human(full)); out.val("lora_model_lora", human(lo)); out.val("lora_model_pct", 100 * lo / full, 3)

# ----------------------------------------------------------- quantisation
nparam = 7e9
for nm, b in (("fp32", 4), ("fp16", 2), ("int8", 1), ("int4", 0.5)):
    out.val(f"mem_{nm}", f"{nparam * b / 1e9:.1f}")
w = rng.normal(0, 0.05, size=(64, 64)); w[0, 0] = 0.4             # one outlier
# [[quant]]
def quantize_int8(w):
    scale = np.abs(w).max() / 127                    # symmetric, per tensor
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale
# [[/quant]]
q, s = quantize_int8(w)
tq = torch.quantize_per_tensor(torch.tensor(w, dtype=torch.float32), float(s), 0, torch.qint8)
out.check("int8 codes vs torch.quantize_per_tensor", q.astype(float), tq.int_repr().numpy().astype(float), atol=0)
deq = q * s
out.val("q_scale", sci(s, 2)); out.val("q_err", sci(np.abs(deq - w).max(), 2))
w2 = w.copy(); w2[0, 0] = 0.05
q2, s2 = quantize_int8(w2)
out.val("q_err_noout", sci(np.abs(q2 * s2 - w2).max(), 2)); out.val("q_ratio", np.abs(deq - w).mean() / np.abs(q2 * s2 - w2).mean(), 1)

# ------------------------------------------------------------ distillation
zt = np.array([4.0, 1.5, 1.0, -1.0]); zs = np.array([3.0, 2.0, 0.0, -0.5]); T = 2.0
pt, ps = softmax(zt / T), softmax(zs / T)
kd = T ** 2 * np.sum(pt * (np.log(pt) - np.log(ps)))
ref = T ** 2 * F.kl_div(F.log_softmax(torch.tensor(zs) / T, 0), F.softmax(torch.tensor(zt) / T, 0), reduction="sum").item()
out.check("distillation loss T^2 KL vs F.kl_div", kd, ref)
out.val("kd", kd, 4); out.val("kd_pt", fv(pt)); out.val("kd_pt1", fv(softmax(zt)))

# ----------------------------------------------------------------- MoE
d_, dff_, E, k_ = 4096, 14336, 8, 2
ffn = 3 * d_ * dff_                                   # gated FFN: three matrices
out.val("moe_ffn", human(ffn)); out.val("moe_total", human(E * ffn)); out.val("moe_active", human(k_ * ffn))
out.val("moe_frac", 100 * k_ / E, 0)

# --------------------------------------------------- compute: C = 6 N D
Np, Dt = 1e9, 20e9
out.val("c_flops", sci(6 * Np * Dt, 1)); out.val("c_tokens", "20")
out.val("c_days", f"{6 * Np * Dt / (100e12 * 0.4) / 86400:.0f}")

# ------------------------------------------------------------ BLEU, ROUGE
ref_s = "the cat sat on the mat".split(); hyp = "the cat is on the mat".split()
def ngram_prec(h, r, n):
    hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
    rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    return sum(min(c, rc[g]) for g, c in hc.items()) / max(1, sum(hc.values()))
ps_ = [ngram_prec(hyp, ref_s, n) for n in (1, 2)]
bp = 1.0 if len(hyp) > len(ref_s) else math.exp(1 - len(ref_s) / len(hyp))
bleu2 = bp * math.exp(np.mean(np.log(ps_)))
out.check("BLEU-2 = BP * geometric mean of precisions", bleu2, bp * gmean(ps_))
out.val("bleu_p1", ps_[0], 3); out.val("bleu_p2", ps_[1], 3); out.val("bleu2", bleu2, 3); out.val("bleu_bp", bp, 1)
def lcs(a, b):
    D = np.zeros((len(a) + 1, len(b) + 1), int)
    for i in range(len(a)):
        for j in range(len(b)):
            D[i + 1, j + 1] = D[i, j] + 1 if a[i] == b[j] else max(D[i, j + 1], D[i + 1, j])
    return D[-1, -1]
l = lcs(hyp, ref_s)
out.val("rouge_lcs", l); out.val("rouge_r", l / len(ref_s), 3); out.val("rouge_p", l / len(hyp), 3)
para = "a cat was sitting on the mat".split()
out.val("bleu_para_p2", ngram_prec(para, ref_s, 2), 3)

# ------------------------------------------------------------- RAG retrieval
docs = ["LoRA adds trainable low-rank matrices to frozen weights.",
        "The KV cache stores keys and values of previous tokens.",
        "BatchNorm normalises each feature over the batch.",
        "Beam search keeps the k most probable partial sequences."]
query = "what does the KV cache store during decoding?"
vec = TfidfVectorizer().fit(docs + [query])
D_ = vec.transform(docs).toarray(); qv = vec.transform([query]).toarray()[0]
cos = D_ @ qv / (np.linalg.norm(D_, axis=1) * np.linalg.norm(qv))
from sklearn.metrics.pairwise import cosine_similarity
out.check("retrieval cosine vs sklearn", cos, cosine_similarity(vec.transform(docs), vec.transform([query])).ravel())
out.val("rag_best", int(np.argmax(cos)) + 1); out.val("rag_score", cos.max(), 3)

# ------------------------------------------------------------ problem values
pa = np.array([0.2, 0.5, 0.1])
out.val("pa_nll", fv(-np.log(pa))); out.val("pa_mean", -np.log(pa).mean(), 3); out.val("pa_ppl", math.exp(-np.log(pa).mean()), 2)
out.check("problem A: PPL = (prod p)^(-1/3)", math.exp(-np.log(pa).mean()), np.prod(pa) ** (-1 / 3))
lb = LoRALinear(nn.Linear(1024, 4096), r=16)
nb = sum(p.numel() for p in lb.parameters() if p.requires_grad)
out.check("problem B LoRA count", nb, 16 * (1024 + 4096), atol=0)
out.val("pb_n", thousands(nb)); out.val("pb_pct", 100 * nb / (1024 * 4096), 2)
zc = np.array([2.0, 1.0, 0.0])
out.val("pc_t05", fv(temperature(zc, 0.5))); out.val("pc_p", fv(softmax(zc))); out.val("pc_topp", fv(top_p(zc, 0.5)))
