def gru_cell(x, h, W, U, bw, bu):
    """PyTorch gate order r, z, n; note r multiplies (U_n h + b_un)."""
    a, bb = W @ x + bw, U @ h + bu
    ar, az, an = np.split(a, 3); br, bz, bn = np.split(bb, 3)
    r, z = sig(ar + br), sig(az + bz)         # reset and update gates
    n = np.tanh(an + r * bn)                  # candidate
    return (1 - z) * n + z * h                # interpolate old and new
