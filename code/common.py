"""Shared helpers for every book script.

Each script calls ``out = setup(__file__)`` and then records values with
``out.val``, data for pgfplots with ``out.dat``, captured text with
``out.text`` and library cross-checks with ``out.check``.  Everything lands
in ``gen/<book>/`` and is \\input into the LaTeX sources, so every number in
the books is produced by code.

Conventions
-----------
* seeds are fixed (numpy, python ``random`` and torch if imported);
* everything runs on CPU in seconds;
* the header of every generated file records library versions.
"""
from __future__ import annotations

import atexit
import importlib
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 0

_LIBS = ["numpy", "scipy", "sklearn", "xgboost", "lightgbm", "torch", "matplotlib"]


def versions() -> dict[str, str]:
    v = {"python": platform.python_version()}
    for name in _LIBS:
        if name in sys.modules or name in ("numpy",):
            try:
                v[name] = importlib.import_module(name).__version__
            except Exception:  # pragma: no cover
                pass
    return v


def version_line() -> str:
    return "versions: " + ", ".join(f"{k} {val}" for k, val in versions().items())


def fmt(x, nd=4):
    """Format a number for LaTeX: ints stay ints, floats get nd significant
    decimals with trailing zeros stripped only when nd is None."""
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):d}"
    x = float(x)
    if np.isnan(x):
        return "NaN"
    s = f"{x:.{nd}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s


def thousands(n: int) -> str:
    """12345678 -> 12{,}345{,}678 for math mode."""
    s = f"{int(n):,}"
    return s.replace(",", "{,}")


class Out:
    def __init__(self, script: Path):
        self.script = Path(script).resolve()
        self.book = self.script.parent.name  # ml or dl
        self.name = self.script.stem
        self.gendir = ROOT / "gen" / self.book
        self.datadir = self.gendir / "data"
        self.txtdir = self.gendir / "out"
        for d in (self.gendir, self.datadir, self.txtdir):
            d.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []
        self.nchecks = 0
        atexit.register(self.close)
        self._closed = False

    # ------------------------------------------------------------------
    def val(self, key: str, value, nd: int = 4, raw: bool = False):
        """Record \\gv{name/key}.  ``raw`` strings are written verbatim."""
        if raw or isinstance(value, str):
            s = str(value)
        else:
            s = fmt(value, nd)
        self.lines.append(f"\\gvset{{{self.name}/{key}}}{{{s}}}")
        return value

    def vals(self, d: dict, nd: int = 4):
        for k, v in d.items():
            self.val(k, v, nd)

    def dat(self, fname: str, cols: dict, nd: int = 6):
        """Write a whitespace-separated .dat file with a header row."""
        keys = list(cols)
        arrs = [np.asarray(cols[k]).ravel() for k in keys]
        n = len(arrs[0])
        assert all(len(a) == n for a in arrs), "columns differ in length"
        path = self.datadir / f"{self.name}-{fname}.dat"
        with open(path, "w") as f:
            f.write(" ".join(keys) + "\n")
            for i in range(n):
                f.write(" ".join(fmt(a[i], nd) if np.isfinite(a[i]) else "nan"
                                 for a in arrs) + "\n")
        return path

    def text(self, fname: str, s: str):
        """Captured program output, shown with \\outputfile in the book."""
        path = self.txtdir / f"{self.name}-{fname}.txt"
        path.write_text(s.rstrip() + "\n")
        return path

    def tex(self, key: str, s: str):
        """A LaTeX fragment (e.g. a tabular body) available as \\gv{name/key}."""
        self.lines.append(f"\\gvset{{{self.name}/{key}}}{{{s}}}")

    def check(self, label: str, ours, ref, atol=1e-6, rtol=1e-5):
        """Assert that our from-scratch result matches the reference library
        and print both.  A mismatch aborts the build."""
        ours_a = np.asarray(ours, dtype=float)
        ref_a = np.asarray(ref, dtype=float)
        ok = ours_a.shape == ref_a.shape and np.allclose(ours_a, ref_a, atol=atol, rtol=rtol)
        diff = float(np.max(np.abs(ours_a - ref_a))) if ours_a.shape == ref_a.shape else float("inf")
        msg = (f"[check] {self.name}: {label}: ours={np.array2string(ours_a.ravel()[:6], precision=6)} "
               f"ref={np.array2string(ref_a.ravel()[:6], precision=6)} max|diff|={diff:.2e} "
               f"-> {'OK' if ok else 'MISMATCH'}")
        print(msg)
        with open(self.gendir / "checks.log", "a") as f:
            f.write(msg + "\n")
        assert ok, msg
        self.nchecks += 1
        return diff

    # ------------------------------------------------------------------
    def close(self):
        if self._closed:
            return
        self._closed = True
        path = self.gendir / f"{self.name}.tex"
        with open(path, "w") as f:
            f.write(f"% generated by code/{self.book}/{self.name}.py -- do not edit\n")
            f.write(f"% {version_line()}\n")
            f.write("\n".join(self.lines) + "\n")


def setup(script, seed: int = SEED) -> Out:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
    out = Out(script)
    print(f"== {out.book}/{out.name} ==  {version_line()}")
    return out
