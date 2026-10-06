def apriori(tx, min_sup):
    items = sorted(set().union(*tx))
    level = [frozenset([i]) for i in items if sup({i}) >= min_sup]
    frequent = list(level)
    while level:
        # join step: unions of two frequent k-sets that differ in one item
        cand = {a | b for a in level for b in level if len(a | b) == len(a) + 1}
        # prune step: every k-subset of a candidate must be frequent (anti-monotonicity)
        cand = {c for c in cand if all(frozenset(s) in set(level) for s in combinations(c, len(c) - 1))}
        level = [c for c in cand if sup(c) >= min_sup]
        frequent += level
    return frequent
