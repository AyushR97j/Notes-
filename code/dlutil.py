"""Small formatting helpers for the Deep Learning book scripts (DL-only)."""
import math


def sci(x: float, nd: int = 2) -> str:
    """LaTeX scientific notation for math mode: 9.54\\times10^{-7}."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(abs(m), nd) >= 10:
        m /= 10
        e += 1
    return f"{m:.{nd}f}\\times10^{{{e}}}"


def human(n: float, nd: int = 1) -> str:
    """Human-readable count: 7.0M, 1.3B, 12.5K (plain text, safe in math)."""
    for unit, s in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= unit:
            return f"{n / unit:.{nd}f}\\,\\mathrm{{{s}}}"
    return f"{n:.0f}"
