def receptive_field(layers):                  # layers: list of (kernel, stride, dilation)
    r, j = 1, 1                               # receptive field, jump (input pixels per step)
    for k, s, d in layers:
        r += (k - 1) * d * j
        j *= s
    return r
