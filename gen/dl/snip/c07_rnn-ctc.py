def collapse(path, blank=0):
    outp, prev = [], None
    for s in path:
        if s != prev and s != blank:            # merge repeats, then drop blanks
            outp.append(s)
        prev = s
    return tuple(outp)
