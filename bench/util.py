"""Small shared utilities (I/O, time formatting, binning, subprocess)."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------ files
def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_csv_rows(path: str | Path, delimiter: str = ",", encoding: str = "utf-8-sig") -> list[dict]:
    with open(path, encoding=encoding, newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter=delimiter)]


def write_csv(path: str | Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------------------------------------------------------------------ zip specs
def make_zip_spec(zip_path: str | Path, member: str) -> str:
    return f"{zip_path}!{member}"


def parse_zip_spec(spec: str) -> tuple[str, str]:
    if "!" not in spec:
        raise ValueError(f"not a zip member spec: {spec}")
    zp, member = spec.split("!", 1)
    return zp, member


# ------------------------------------------------------------------------------------ time / bins
def fmt_time(t_s: Optional[float], decimals: int = 1) -> str:
    """Seconds -> 'MM:SS.s' (or 'H:MM:SS.s' for >= 1 h)."""
    if t_s is None or (isinstance(t_s, float) and math.isnan(t_s)):
        return "?"
    t = max(0.0, float(t_s))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    sec = f"{s:0{3 + decimals}.{decimals}f}" if decimals else f"{int(round(s)):02d}"
    return f"{h}:{m:02d}:{sec}" if h else f"{m:02d}:{sec}"


def fmt_duration(t_s: Optional[float]) -> str:
    if t_s is None:
        return "unknown"
    t = float(t_s)
    if t < 60:
        return f"{t:.1f} s"
    return f"{int(t // 60)} min {int(round(t % 60)):02d} s"


def duration_bin(d_s: Optional[float], bins_s: Sequence[float]) -> str:
    """Bin a duration with cut points in seconds; labels in minutes, e.g. '<5min', '5-8min', '>12min'."""
    if d_s is None:
        return "unknown"
    mins = [b / 60.0 for b in bins_s]
    labels = []
    for i, b in enumerate(mins):
        lo = mins[i - 1] if i else None
        labels.append(f"<{b:g}min" if lo is None else f"{lo:g}-{b:g}min")
    labels.append(f">{mins[-1]:g}min" if mins else "all")
    for i, b in enumerate(bins_s):
        if d_s < b:
            return labels[i]
    return labels[-1]


def human_size(n: Optional[float]) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


# ------------------------------------------------------------------------------------ text
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_simple(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def chunked(seq: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ------------------------------------------------------------------------------------ subprocess
class CmdResult:
    def __init__(self, rc: int, out: str, err: str, cmd: Sequence[str]):
        self.rc, self.out, self.err, self.cmd = rc, out, err, list(cmd)

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def __repr__(self) -> str:
        return f"CmdResult(rc={self.rc}, cmd={' '.join(self.cmd)[:200]})"


def run_cmd(cmd: Sequence[str], timeout: Optional[float] = None, cwd: Optional[str | Path] = None,
            input_text: Optional[str] = None) -> CmdResult:
    """Run a command, capture text output; never raises on non-zero exit (check .ok)."""
    try:
        p = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, cwd=cwd, input=input_text)
        return CmdResult(p.returncode, p.stdout, p.stderr, cmd)
    except subprocess.TimeoutExpired as e:
        return CmdResult(124, e.stdout or "" if isinstance(e.stdout, str) else "", f"timeout after {timeout}s", cmd)
    except FileNotFoundError as e:
        return CmdResult(127, "", str(e), cmd)
