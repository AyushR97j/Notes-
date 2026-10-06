def beam_search(k, T, start=0):
    beams = [((), 0.0)]                         # (tokens, total log-prob)
    for t in range(T):
        cand = []
        for seq, s in beams:
            prev = seq[-1] if seq else start
            lp = logp(prev, t)
            cand += [(seq + (v,), s + lp[v]) for v in range(V)]
        beams = sorted(cand, key=lambda c: -c[1])[:k]   # keep the k best prefixes
    return beams[0]
