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
