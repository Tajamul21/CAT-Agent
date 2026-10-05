"""Media utilities for the prepare stage (DESIGN.md section 6).

Everything here shells out to the ffmpeg CLI (the imageio-ffmpeg static binary) and, when one is
configured/available, an external ``ffprobe``; no opencv/av/decord.  Each ffmpeg invocation goes
through :func:`run_ffmpeg`, which records the exact command line on the logger it is given, so
per-sample ``prepare.log`` files are reproducible by copy-paste.

Public API
----------
* :func:`ffprobe` - duration/fps/size/codec of a video (ffprobe JSON, else ``ffmpeg -i`` banner).
* :func:`materialize_source` - make ``<sample_dir>/source.mp4`` for ``file`` (symlink),
  ``zip_member`` (cached extraction + symlink) and ``concat_clips`` (concat demuxer) records.
* :func:`plan_frame_times` - grid + boundary + midpoint frame schedule (pure function).
* :func:`extract_frame`, :func:`overlay_text`, :func:`build_contact_sheet`, :func:`make_preview`.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from bench.config import DEFAULTS, Config, load_config
from bench.schema import ProbeInfo, Segment, VideoRecord
from bench.util import CmdResult, parse_zip_spec, run_cmd, write_json

# ------------------------------------------------------------------------------------ constants
FFMPEG_THREADS = 2            # per job; the prepare stage runs many jobs in parallel
DEDUPE_WINDOW_S = 1.5         # frames closer than this are considered duplicates (long videos)
END_GUARD_S = 0.25            # never seek into the final quarter second (decoders may return nothing)
PROBE_TIMEOUT_S = 120
FRAME_TIMEOUT_S = 180
CONCAT_TIMEOUT_S = 3600
PREVIEW_TIMEOUT_S = 7200
SOURCE_NAME = "source.mp4"
CONCAT_LIST_NAME = "concat_list.txt"
CONCAT_MAP_NAME = "concat_map.json"

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)

_EXTRACT_LOCKS: dict[str, threading.Lock] = {}
_EXTRACT_LOCKS_GUARD = threading.Lock()


class MediaError(RuntimeError):
    """A media operation failed (missing input, ffmpeg error, unreadable or empty output)."""


# ------------------------------------------------------------------------------------ helpers
def _cfg(cfg: Optional[Config]) -> Config:
    return cfg if cfg is not None else load_config()


def _log(log: Optional[logging.Logger], level: int, msg: str, *args: Any) -> None:
    if log is not None:
        log.log(level, msg, *args)


def run_ffmpeg(cmd: Sequence[str], log: Optional[logging.Logger] = None,
               timeout: Optional[float] = None) -> CmdResult:
    """Run an ffmpeg/ffprobe command, logging the exact command line and its outcome."""
    cmd = [str(c) for c in cmd]
    _log(log, logging.INFO, "$ %s", shlex.join(cmd))
    t0 = time.time()
    res = run_cmd(cmd, timeout=timeout)
    tail = (res.err or "").strip().splitlines()
    tail_s = tail[-1][:300] if tail else ""
    if res.ok:
        _log(log, logging.DEBUG, "  -> rc=0 in %.1fs%s", time.time() - t0, f" ({tail_s})" if tail_s else "")
    else:
        _log(log, logging.WARNING, "  -> rc=%s in %.1fs: %s", res.rc, time.time() - t0, tail_s)
    return res


def _parse_fraction(s: Any) -> Optional[float]:
    """'2997/100' -> 29.97; '25' -> 25.0; '0/0' or junk -> None."""
    if s is None:
        return None
    try:
        s = str(s).strip()
        if "/" in s:
            num, den = s.split("/", 1)
            num_f, den_f = float(num), float(den)
            if den_f == 0 or num_f <= 0:
                return None
            return num_f / den_f
        v = float(s)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _to_float(s: Any) -> Optional[float]:
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _to_int(s: Any) -> Optional[int]:
    try:
        return int(str(s))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------ probing
_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_VIDEO_LINE_RE = re.compile(r"Stream #\d+:\d+.*?Video:\s*([A-Za-z0-9_]+)")
_SIZE_RE = re.compile(r"(?<![0-9A-Za-z])(\d{2,5})x(\d{2,5})(?![0-9A-Za-z])")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_TBR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*tbr")


def parse_ffmpeg_banner(text: str) -> ProbeInfo:
    """Parse the ``ffmpeg -i <file>`` stderr banner (fallback when no ffprobe is available).

    Returns a ProbeInfo whose ``duration_s`` is 0.0 when no Duration line was found.
    """
    info = ProbeInfo()
    m = _DUR_RE.search(text or "")
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info.duration_s = h * 3600 + mi * 60 + s
    for line in (text or "").splitlines():
        vm = _VIDEO_LINE_RE.search(line)
        if not vm:
            continue
        info.codec = vm.group(1)
        sm = _SIZE_RE.search(line)
        if sm:
            info.width, info.height = int(sm.group(1)), int(sm.group(2))
        fm = _FPS_RE.search(line) or _TBR_RE.search(line)
        if fm:
            info.fps = _to_float(fm.group(1))
        break
    return info


def _probe_with_ffprobe(exe: str, path: Path, log: Optional[logging.Logger]) -> Optional[ProbeInfo]:
    cmd = [exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    res = run_ffmpeg(cmd, log, timeout=PROBE_TIMEOUT_S)
    if not res.ok or not res.out.strip():
        return None
    try:
        data = json.loads(res.out)
    except json.JSONDecodeError:
        return None
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        return None
    v = streams[0]
    fmt = data.get("format", {}) or {}
    duration = _to_float(fmt.get("duration")) or _to_float(v.get("duration")) or 0.0
    fps = _parse_fraction(v.get("avg_frame_rate")) or _parse_fraction(v.get("r_frame_rate"))
    return ProbeInfo(
        duration_s=float(duration),
        fps=fps,
        width=_to_int(v.get("width")),
        height=_to_int(v.get("height")),
        codec=v.get("codec_name"),
        nb_frames=_to_int(v.get("nb_frames")),
        size_bytes=_to_int(fmt.get("size")),
    )


def _probe_with_ffmpeg(exe: str, path: Path, log: Optional[logging.Logger]) -> ProbeInfo:
    res = run_ffmpeg([exe, "-hide_banner", "-nostdin", "-i", str(path)], log, timeout=PROBE_TIMEOUT_S)
    # ffmpeg -i without output exits non-zero by design; the banner is on stderr.
    return parse_ffmpeg_banner(res.err)


def ffprobe(path: str | Path, cfg: Optional[Config] = None, log: Optional[logging.Logger] = None) -> ProbeInfo:
    """Probe a video. Uses ``cfg.ffprobe()`` when available, else parses ``ffmpeg -i`` stderr.

    Raises :class:`MediaError` if the file is missing or no positive duration can be determined.
    """
    cfg = _cfg(cfg)
    p = Path(path)
    if not p.exists():
        raise MediaError(f"file not found: {p}")
    size = p.stat().st_size
    info: Optional[ProbeInfo] = None
    exe = cfg.ffprobe()
    if exe:
        info = _probe_with_ffprobe(exe, p, log)
    if info is None or info.duration_s <= 0:
        fb = _probe_with_ffmpeg(cfg.ffmpeg(), p, log)
        if info is None:
            info = fb
        else:  # keep what ffprobe gave us, fill the gaps from the banner
            info.duration_s = info.duration_s or fb.duration_s
            info.fps = info.fps or fb.fps
            info.width = info.width or fb.width
            info.height = info.height or fb.height
            info.codec = info.codec or fb.codec
    if info.duration_s <= 0:
        raise MediaError(f"could not determine duration of {p}")
    info.size_bytes = size
    return info


# ------------------------------------------------------------------------------------ sources
def _resolve_source(path_s: str, cfg: Config) -> Path:
    p = Path(path_s)
    if not p.is_absolute():
        p = Path(cfg.paths.datasets_root) / p
    return p


def _link(target: Path, dst: Path, log: Optional[logging.Logger]) -> Path:
    """Create/refresh symlink ``dst`` -> ``target`` (absolute)."""
    target = target.resolve()
    if not target.exists():
        raise MediaError(f"source file not found: {target}")
    if dst.is_symlink():
        try:
            if Path(os.readlink(dst)) == target:
                _log(log, logging.INFO, "source link already in place: %s -> %s", dst.name, target)
                return dst
        except OSError:
            pass
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    dst.symlink_to(target)
    _log(log, logging.INFO, "linked %s -> %s", dst.name, target)
    return dst


def _safe_member_path(member: str) -> Path:
    parts = [p for p in Path(member.replace("\\", "/")).parts if p not in ("", "/", "..", ".")]
    if not parts:
        raise MediaError(f"invalid zip member name: {member!r}")
    return Path(*parts)


def _lock_for(key: Path) -> threading.Lock:
    with _EXTRACT_LOCKS_GUARD:
        return _EXTRACT_LOCKS.setdefault(str(key), threading.Lock())


def extract_zip_member(spec: str, cache_root: str | Path, log: Optional[logging.Logger] = None) -> Path:
    """Extract ``<zip>!<member>`` into ``cache_root/<member>`` unless it is already there with the right size."""
    zip_path, member = parse_zip_spec(spec)
    zp = Path(zip_path)
    if not zp.exists():
        raise MediaError(f"zip archive not found: {zp}")
    target = Path(cache_root) / _safe_member_path(member)
    with _lock_for(target):
        with zipfile.ZipFile(zp) as zf:
            try:
                info = zf.getinfo(member)
            except KeyError:
                raise MediaError(f"member {member!r} not found in {zp}") from None
            if target.exists() and target.stat().st_size == info.file_size:
                _log(log, logging.INFO, "zip member cached: %s (%d bytes)", target, info.file_size)
                return target
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + f".part{os.getpid()}")
            _log(log, logging.INFO, "extracting %s!%s -> %s (%d bytes)", zp, member, target, info.file_size)
            t0 = time.time()
            try:
                with zf.open(info) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 4 << 20)
                os.replace(tmp, target)
            finally:
                if tmp.exists():
                    tmp.unlink()
            size = target.stat().st_size
            if size != info.file_size:
                raise MediaError(f"extracted size mismatch for {member}: {size} != {info.file_size}")
            _log(log, logging.INFO, "extracted in %.1fs", time.time() - t0)
    return target


def _duration_close(actual: float, expected: float, n_clips: int) -> bool:
    """Concat output duration vs sum of clip durations; allow 1 s or 1% + 50 ms per clip."""
    tol = max(1.0, 0.01 * expected + 0.05 * n_clips)
    return abs(actual - expected) <= tol


def _concat_escape(path: str) -> str:
    return path.replace("'", r"'\''")


def concat_clips(clip_paths: Sequence[str | Path], sample_dir: str | Path, cfg: Optional[Config] = None,
                 log: Optional[logging.Logger] = None, force: bool = False) -> Path:
    """Concatenate clips into ``<sample_dir>/source.mp4`` (stream copy, re-encode fallback).

    Writes ``concat_list.txt`` and ``concat_map.json`` (``[{clip, offset_s, duration_s}]`` with offsets
    from probing each clip).  An existing output whose duration matches is reused unless ``force``.
    """
    cfg = _cfg(cfg)
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    clips = [_resolve_source(str(p), cfg) for p in clip_paths]
    if not clips:
        raise MediaError("concat_clips record has no source_paths")
    missing = [str(c) for c in clips if not c.exists()]
    if missing:
        raise MediaError(f"{len(missing)} of {len(clips)} clips missing, first: {missing[0]}")

    dst = sample_dir / SOURCE_NAME
    list_path = sample_dir / CONCAT_LIST_NAME
    map_path = sample_dir / CONCAT_MAP_NAME

    concat_map: list[dict] = []
    offset = 0.0
    for c in clips:
        pi = ffprobe(c, cfg, log)
        concat_map.append({"clip": str(c), "offset_s": round(offset, 3), "duration_s": round(pi.duration_s, 3)})
        offset += pi.duration_s
    expected = offset

    if not force and dst.exists() and not dst.is_symlink() and map_path.exists():
        try:
            pi = ffprobe(dst, cfg, log)
            if _duration_close(pi.duration_s, expected, len(clips)):
                _log(log, logging.INFO, "reusing existing concat output (%.2fs vs expected %.2fs)", pi.duration_s, expected)
                write_json(map_path, concat_map)
                return dst
            _log(log, logging.WARNING, "existing concat output duration %.2fs != expected %.2fs; rebuilding",
                 pi.duration_s, expected)
        except MediaError as e:
            _log(log, logging.WARNING, "existing concat output unreadable (%s); rebuilding", e)

    list_path.write_text("".join(f"file '{_concat_escape(str(c))}'\n" for c in clips), encoding="utf-8")
    ffmpeg = cfg.ffmpeg()
    tmp = sample_dir / "source.part.mp4"
    if tmp.exists():
        tmp.unlink()
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", str(FFMPEG_THREADS),
            "-f", "concat", "-safe", "0", "-i", str(list_path)]
    tail = ["-an", "-movflags", "+faststart", "-threads", str(FFMPEG_THREADS), str(tmp)]

    res = run_ffmpeg(base + ["-c", "copy"] + tail, log, timeout=CONCAT_TIMEOUT_S)
    ok = res.ok and tmp.exists() and tmp.stat().st_size > 0
    reason = "" if ok else f"rc={res.rc}"
    if ok:
        try:
            pi = ffprobe(tmp, cfg, log)
            if not _duration_close(pi.duration_s, expected, len(clips)):
                ok, reason = False, f"duration {pi.duration_s:.2f}s != expected {expected:.2f}s"
        except MediaError as e:
            ok, reason = False, str(e)
    if not ok:
        _log(log, logging.WARNING, "concat with -c copy failed (%s); falling back to re-encode", reason)
        if tmp.exists():
            tmp.unlink()
        enc = ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        res = run_ffmpeg(base + enc + tail, log, timeout=CONCAT_TIMEOUT_S)
        if not res.ok or not tmp.exists() or tmp.stat().st_size == 0:
            raise MediaError(f"concat re-encode failed: rc={res.rc} {res.err.strip()[-300:]}")
        pi = ffprobe(tmp, cfg, log)
        if not _duration_close(pi.duration_s, expected, len(clips)):
            _log(log, logging.WARNING, "re-encoded concat duration %.2fs differs from expected %.2fs",
                 pi.duration_s, expected)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.replace(tmp, dst)
    write_json(map_path, concat_map)
    _log(log, logging.INFO, "concatenated %d clips -> %s (expected %.2fs)", len(clips), dst.name, expected)
    return dst


def materialize_source(record: VideoRecord, sample_dir: str | Path, cfg: Optional[Config] = None,
                       log: Optional[logging.Logger] = None, force: bool = False) -> Path:
    """Create ``<sample_dir>/source.mp4`` for the record and return its path.

    * ``file``        -> symlink to the absolute source path.
    * ``zip_member``  -> extract into ``data/cache/extracted/<dataset>/<member>`` (skipped when the cached
      copy has the right size), then symlink.
    * ``concat_clips`` -> :func:`concat_clips` (writes concat_list.txt + concat_map.json).
    """
    cfg = _cfg(cfg)
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    dst = sample_dir / SOURCE_NAME
    kind = record.source_kind
    if not record.source_paths:
        raise MediaError(f"{record.sample_id}: record has no source_paths")
    if kind == "file":
        return _link(_resolve_source(record.source_paths[0], cfg), dst, log)
    if kind == "zip_member":
        cache_root = Path(cfg.paths.cache) / "extracted" / (record.dataset or "unknown")
        extracted = extract_zip_member(record.source_paths[0], cache_root, log)
        return _link(extracted, dst, log)
    if kind == "concat_clips":
        return concat_clips(record.source_paths, sample_dir, cfg, log, force=force)
    raise MediaError(f"{record.sample_id}: unknown source_kind {kind!r}")


# ------------------------------------------------------------------------------------ frame planning
@dataclass
class _Cand:
    t: float
    reason: str      # grid | boundary | midpoint
    prio: int        # 0 boundary, 1 midpoint, 2 grid (lower wins on dedupe)
    seg_len: float   # segment length for boundary/midpoint, inf for grid


def _frames_cfg(media_cfg: Optional[dict]) -> dict:
    base = dict(DEFAULTS["media"]["frames"])
    if not media_cfg:
        return base
    sub = media_cfg.get("frames") if isinstance(media_cfg.get("frames"), dict) else media_cfg
    base.update({k: v for k, v in sub.items() if v is not None})
    return base


def _seg_bounds(seg: Any) -> tuple[Optional[float], Optional[float]]:
    if isinstance(seg, Segment):
        return _to_float(seg.start_s), _to_float(seg.end_s)
    if isinstance(seg, dict):
        return _to_float(seg.get("start_s")), _to_float(seg.get("end_s"))
    if isinstance(seg, (tuple, list)) and len(seg) >= 2:
        return _to_float(seg[0]), _to_float(seg[1])
    return None, None


def plan_frame_times(duration_s: Optional[float], segments: Optional[Sequence[Any]] = None,
                     media_cfg: Optional[dict] = None) -> list[tuple[float, str]]:
    """Plan frame timestamps for a video (pure function, deterministic).

    ``n = clamp(round(duration / seconds_per_frame), min_frames, max_frames)`` uniform grid points at
    ``(i + 0.5) * duration / n``; when ``segments`` are given (and ``boundary_frames`` is on) also each
    segment start + 1.0 s ("boundary") and each segment midpoint ("midpoint").  Candidates are
    de-duplicated within ``min(1.5 s, 0.45 * grid spacing)`` (boundary > midpoint > grid), then trimmed to
    ``max_frames`` by dropping grid points nearest to boundary/midpoint frames (then midpoints and
    boundaries of the shortest segments).  Returns ``[(t_s, reason)]`` sorted by time.
    """
    fc = _frames_cfg(media_cfg)
    spf = float(fc.get("seconds_per_frame") or 10)
    n_min = int(fc.get("min_frames") or 1)
    n_max = int(fc.get("max_frames") or 48)
    use_boundaries = bool(fc.get("boundary_frames", True))
    dur = _to_float(duration_s) or 0.0
    if dur <= 0 or n_max <= 0:
        return []
    n_min = max(1, min(n_min, n_max))
    n = int(round(dur / spf)) if spf > 0 else n_min
    n = max(n_min, min(n_max, n))
    spacing = dur / n
    window = min(DEDUPE_WINDOW_S, 0.45 * spacing)
    t_max = max(0.0, dur - min(END_GUARD_S, dur * 0.05))

    def clamp(t: float) -> float:
        return min(max(0.0, t), t_max)

    cands: list[_Cand] = [_Cand(clamp((i + 0.5) * spacing), "grid", 2, math.inf) for i in range(n)]
    if use_boundaries and segments:
        for seg in segments:
            s, e = _seg_bounds(seg)
            if s is None:
                continue
            s = min(max(0.0, s), dur)
            e = min(max(s, e if e is not None else s), dur)
            length = e - s
            boundary = s + (min(1.0, length / 2.0) if length > 0 else 0.0)
            cands.append(_Cand(clamp(boundary), "boundary", 0, length))
            if length > 0:
                cands.append(_Cand(clamp((s + e) / 2.0), "midpoint", 1, length))

    accepted: list[_Cand] = []
    for c in sorted(cands, key=lambda c: (c.prio, c.t)):
        if all(abs(c.t - a.t) >= window for a in accepted):
            accepted.append(c)

    while len(accepted) > n_max:
        grid = [c for c in accepted if c.reason == "grid"]
        anchors = [a.t for a in accepted if a.reason != "grid"]
        if grid and anchors:
            victim = min(grid, key=lambda g: (min(abs(g.t - a) for a in anchors), g.t))
        elif grid:
            victim = grid[-1]
        else:
            pool = [c for c in accepted if c.reason == "midpoint"] or accepted
            victim = min(pool, key=lambda c: (c.seg_len, -c.t))
        accepted.remove(victim)

    accepted.sort(key=lambda c: c.t)
    return [(round(c.t, 3), c.reason) for c in accepted]


# ------------------------------------------------------------------------------------ frames
def _scale_expr(long_side: int) -> str:
    L = int(long_side)
    return f"scale='if(gte(iw,ih),min({L},iw),-2)':'if(gte(iw,ih),-2,min({L},ih))'"


def extract_frame(src: str | Path, t_s: float, out_path: str | Path, cfg: Optional[Config] = None,
                  long_side: int = 768, log: Optional[logging.Logger] = None) -> Path:
    """Extract one JPEG frame at ``t_s`` using input seeking; long side limited to ``long_side`` px."""
    cfg = _cfg(cfg)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    cmd = [cfg.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-threads", str(FFMPEG_THREADS), "-ss", f"{max(0.0, float(t_s)):.3f}", "-i", str(src),
           "-frames:v", "1", "-update", "1", "-vf", _scale_expr(long_side), "-q:v", "3",
           "-threads", str(FFMPEG_THREADS), str(out)]
    res = run_ffmpeg(cmd, log, timeout=FRAME_TIMEOUT_S)
    if not res.ok:
        raise MediaError(f"ffmpeg frame extraction failed at t={t_s:.2f}s (rc={res.rc}): {res.err.strip()[-300:]}")
    if not out.exists() or out.stat().st_size == 0:
        raise MediaError(f"no frame decoded at t={t_s:.2f}s (past end of stream?)")
    return out


@lru_cache(maxsize=1)
def find_font_path() -> Optional[str]:
    """DejaVuSans.ttf under /usr/share/fonts if present, else None (PIL default font is used)."""
    for cand in _FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    root = Path("/usr/share/fonts")
    if root.exists():
        for dirpath, _dirs, files in os.walk(root):
            if "DejaVuSans.ttf" in files:
                return str(Path(dirpath) / "DejaVuSans.ttf")
    return None


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    path = find_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def draw_overlay(img: Image.Image, lines: Sequence[str], font_size: Optional[int] = None,
                 pad: int = 6, margin: int = 8) -> Image.Image:
    """Draw a black box with white text lines at the bottom-left of ``img`` (in place)."""
    lines = [str(l) for l in lines if l]
    if not lines:
        return img
    w, h = img.size
    fs = int(font_size or max(11, min(28, round(w / 34))))
    draw = ImageDraw.Draw(img)
    for _attempt in range(3):
        font = _load_font(fs)
        boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
        text_w = max(b[2] for b in boxes)
        line_h = max(b[3] for b in boxes) + 2
        box_w = text_w + 2 * pad
        if box_w <= w - 2 * margin or fs <= 9:
            break
        fs = max(9, int(fs * 0.8))
    box_h = len(lines) * line_h + 2 * pad
    x0, y1 = margin, h - margin
    y0 = max(0, y1 - box_h)
    x1 = min(w - 1, x0 + box_w)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    y = y0 + pad
    for ln in lines:
        draw.text((x0 + pad, y), ln, font=font, fill=(255, 255, 255))
        y += line_h
    return img


def overlay_text(jpg_path: str | Path, lines: Sequence[str], out_path: Optional[str | Path] = None,
                 quality: int = 85) -> Path:
    """Burn ``lines`` into the bottom-left of a JPEG (PIL); writes ``out_path`` or overwrites in place."""
    src = Path(jpg_path)
    dst = Path(out_path) if out_path else src
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        img = im.convert("RGB")
    draw_overlay(img, lines)
    img.save(dst, "JPEG", quality=int(quality), optimize=True)
    return dst


def build_contact_sheet(frame_paths: Sequence[str | Path], out_path: str | Path, cols: int = 6,
                        tile_width: int = 320, labels: Optional[Sequence[Sequence[str]]] = None,
                        quality: int = 85) -> Path:
    """Tile frames into a grid JPEG. ``labels[i]`` (optional) is overlaid on tile i at tile scale."""
    paths = [Path(p) for p in frame_paths]
    if not paths:
        raise MediaError("no frames for contact sheet")
    cols = max(1, min(int(cols), len(paths)))
    rows = math.ceil(len(paths) / cols)
    tiles: list[Image.Image] = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            img = im.convert("RGB")
        w, h = img.size
        th = max(1, round(tile_width * h / max(1, w)))
        img = img.resize((int(tile_width), th), Image.LANCZOS)
        if labels is not None and i < len(labels) and labels[i]:
            draw_overlay(img, labels[i])
        tiles.append(img)
    tile_h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * int(tile_width), rows * tile_h), (24, 24, 24))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (c * int(tile_width), r * tile_h))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=int(quality), optimize=True)
    return out


# ------------------------------------------------------------------------------------ preview
def make_preview(src: str | Path, out_path: str | Path, height: int = 360, crf: int = 28,
                 max_duration_s: Optional[float] = None, cfg: Optional[Config] = None,
                 log: Optional[logging.Logger] = None) -> Path:
    """Encode a small h264 preview (no audio, faststart). Never upscales beyond the source height."""
    cfg = _cfg(cfg)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    if tmp.exists():
        tmp.unlink()
    cmd = [cfg.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-threads", str(FFMPEG_THREADS), "-i", str(src)]
    if max_duration_s:
        cmd += ["-t", f"{float(max_duration_s):.3f}"]
    cmd += ["-vf", f"scale=-2:'min({int(height)},ih)'", "-c:v", "libx264", "-crf", str(int(crf)),
            "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
            "-threads", str(FFMPEG_THREADS), str(tmp)]
    res = run_ffmpeg(cmd, log, timeout=PREVIEW_TIMEOUT_S)
    if not res.ok or not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            tmp.unlink()
        raise MediaError(f"preview encode failed (rc={res.rc}): {res.err.strip()[-300:]}")
    os.replace(tmp, out)
    return out
