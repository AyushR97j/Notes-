def enc_layer_params(d, dff):
    attn = 4 * d * d + 4 * d                 # W_Q, W_K, W_V, W_O and their biases
    ffn = d * dff + dff + dff * d + d        # two linear layers
    ln = 2 * (2 * d)                         # two LayerNorms (gamma, beta)
    return attn + ffn + ln

def dec_layer_params(d, dff):
    return enc_layer_params(d, dff) + (4 * d * d + 4 * d) + 2 * d   # + cross-attn + 3rd LN
