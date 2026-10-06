def quantize_int8(w):
    scale = np.abs(w).max() / 127                    # symmetric, per tensor
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale
