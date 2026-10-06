"""Extract code regions marked  # [[name]] ... # [[/name]]  from
code/<book>/*.py into gen/<book>/snip/<script>-<name>.py so the book can
show exactly the code that produced its numbers."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
book = sys.argv[1]
outdir = ROOT / "gen" / book / "snip"
outdir.mkdir(parents=True, exist_ok=True)
start = re.compile(r"^\s*# \[\[([\w-]+)\]\]\s*$")
for py in sorted((ROOT / "code" / book).glob("*.py")):
    lines = py.read_text().splitlines()
    open_name, buf, indent = None, [], 0
    for ln in lines:
        m = start.match(ln)
        if m and open_name is None:
            open_name, buf = m.group(1), []
            indent = len(ln) - len(ln.lstrip())
            continue
        if open_name and ln.strip() == f"# [[/{open_name}]]":
            text = "\n".join(l[indent:] if l[:indent].strip() == "" else l for l in buf)
            (outdir / f"{py.stem}-{open_name}.py").write_text(text.rstrip() + "\n")
            open_name = None
            continue
        if open_name:
            buf.append(ln)
    assert open_name is None, f"unclosed snippet {open_name} in {py}"
