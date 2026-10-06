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
