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
