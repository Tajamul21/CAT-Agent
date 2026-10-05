"""Inventory stage (DESIGN.md section 4).

For every dataset adapter: enumerate the candidate videos, verify that each source actually exists
(plain files, zip members, concat clip lists), optionally fill unknown durations with ffprobe, and
write

* ``data/inventory/<dataset>.jsonl``          one ``VideoRecord`` per line (verified records only)
* ``data/inventory/<dataset>.missing.jsonl``  records whose media could not be found (debugging aid)
* ``data/inventory/summary.json`` / ``summary.md``   per-dataset statistics (adapter.stats + checks)
* ``data/cache/probe_cache.jsonl``            ffprobe results keyed by ``path|size`` (reused across runs)

Usage: ``./ophbench inventory [--datasets a,b] [--probe-durations] [--workers N]``.

Adapters are imported lazily; one that fails to import (or raises while iterating) is logged as an
ERROR and skipped so the remaining datasets are still inventoried.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from bench.config import Config
from bench.datasets import ADAPTER_SPECS, get_adapter_class
from bench.log import JsonlWriter, get_logger, now_iso, record_run, redact
from bench.schema import DATASETS, VideoRecord
from bench.util import duration_bin, md_table, parse_zip_spec, read_json, run_cmd, write_json, write_jsonl

STAGE = "inventory"
#: directories referenced by at least this many source paths are listed once with os.scandir
#: instead of being stat()ed path by path (the Ophora clips folder has 162k entries).
SCANDIR_THRESHOLD = 8
PROBE_TIMEOUT_S = 120
#: per-dataset cap on individual WARNING lines for missing media / probe failures
MAX_ITEM_WARNINGS = 20
#: label keys with more distinct values than this are left out of the label histograms
MAX_LABEL_CARDINALITY = 20


# ------------------------------------------------------------------------------ path checks
class PathChecker:
    """Existence checks with a per-directory listing cache and a per-archive namelist cache.

    Call :meth:`prime` with every file path first so that "hot" directories are listed once with
    ``os.scandir``; everything else falls back to ``os.path.exists``.
    """

    def __init__(self, log: logging.Logger):
        self.log = log
        self._dir_counts: Counter[str] = Counter()
        self._dir_cache: dict[str, set[str]] = {}
        self._zip_cache: dict[str, Optional[set[str]]] = {}
        self.n_scandir = 0
        self.n_stat = 0
        self.n_zip_opened = 0

    def prime(self, paths: Iterable[str]) -> None:
        """Count how many paths reference each directory (decides scandir vs stat)."""
        for p in paths:
            self._dir_counts[os.path.dirname(p)] += 1

    def file_exists(self, path: str) -> bool:
        d, name = os.path.split(path)
        if self._dir_counts.get(d, 0) >= SCANDIR_THRESHOLD:
            names = self._dir_cache.get(d)
            if names is None:
                names = self._scan(d)
                self._dir_cache[d] = names
            return name in names
        self.n_stat += 1
        return os.path.exists(path)

    def _scan(self, d: str) -> set[str]:
        self.n_scandir += 1
        try:
            with os.scandir(d) as it:
                return {e.name for e in it}
        except OSError as e:
            self.log.warning("cannot list directory %s: %s", d, e)
            return set()

    def zip_member_exists(self, spec: str) -> tuple[bool, str]:
        """Check a ``<zip path>!<member>`` spec; returns (ok, problem description)."""
        try:
            zp, member = parse_zip_spec(spec)
        except ValueError as e:
            return False, str(e)
        names = self._zip_names(zp)
        if names is None:
            return False, f"archive missing or unreadable: {zp}"
        if member in names:
            return True, ""
        alt = sorted(n for n in names if n.endswith("/" + member))
        hint = f" (found as '{alt[0]}')" if alt else ""
        return False, f"member '{member}' not in {os.path.basename(zp)}{hint}"

    def _zip_names(self, zp: str) -> Optional[set[str]]:
        if zp in self._zip_cache:
            return self._zip_cache[zp]
        names: Optional[set[str]] = None
        try:
            with zipfile.ZipFile(zp) as zf:
                names = set(zf.namelist())
            self.n_zip_opened += 1
        except (OSError, zipfile.BadZipFile) as e:
            self.log.warning("cannot read zip %s: %s", zp, e)
        self._zip_cache[zp] = names
        return names


def verify_record(rec: VideoRecord, checker: PathChecker) -> list[str]:
    """Return a list of problems with the record's sources (empty list = all media present)."""
    if not rec.source_paths:
        return ["no source_paths"]
    if rec.source_kind == "file":
        return [f"missing file: {p}" for p in rec.source_paths if not checker.file_exists(p)]
    if rec.source_kind == "zip_member":
        problems = []
        for spec in rec.source_paths:
            ok, why = checker.zip_member_exists(spec)
            if not ok:
                problems.append(f"missing zip member: {why}")
        return problems
    if rec.source_kind == "concat_clips":
        missing = [p for p in rec.source_paths if not checker.file_exists(p)]
        if missing:
            return [f"missing {len(missing)}/{len(rec.source_paths)} clips (first: {missing[0]})"]
        return []
    return [f"unknown source_kind '{rec.source_kind}'"]


# ------------------------------------------------------------------------------ probing
_FRACTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?\s*$")
_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RES_RE = re.compile(r"Video:.*?[\s,](\d{2,5})x(\d{2,5})(?:[\s,\[]|$)")
_FPS_RE = re.compile(r"([\d.]+)\s*fps")


def parse_rate(s: Any) -> Optional[float]:
    """'30000/1001' -> 29.97; '60/1' -> 60.0; None/garbage -> None."""
    if s is None:
        return None
    m = _FRACTION_RE.match(str(s))
    if not m:
        return None
    num = float(m.group(1))
    den = float(m.group(2)) if m.group(2) else 1.0
    if den == 0 or num == 0:
        return None
    return round(num / den, 3)


def _empty_probe() -> dict:
    return {"ok": False, "duration_s": None, "fps": None, "width": None, "height": None, "error": None, "tool": None}


def parse_ffprobe_json(text: str) -> dict:
    """Parse ``ffprobe -show_entries format=duration:stream=r_frame_rate,width,height -of json``."""
    data = json.loads(text)
    out = _empty_probe()
    fmt = data.get("format") or {}
    try:
        out["duration_s"] = round(float(fmt.get("duration")), 3)
    except (TypeError, ValueError):
        pass
    for st in data.get("streams") or []:
        if st.get("width") and st.get("height"):
            out["width"], out["height"] = int(st["width"]), int(st["height"])
            out["fps"] = parse_rate(st.get("r_frame_rate") or st.get("avg_frame_rate"))
            break
    out["ok"] = out["duration_s"] is not None
    out["tool"] = "ffprobe"
    return out


def parse_ffmpeg_stderr(text: str) -> dict:
    """Fallback parser for ``ffmpeg -i <file>`` stderr when no ffprobe binary is available."""
    out = _empty_probe()
    m = _DUR_RE.search(text or "")
    if m:
        out["duration_s"] = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 3)
    m = _RES_RE.search(text or "")
    if m:
        out["width"], out["height"] = int(m.group(1)), int(m.group(2))
    m = _FPS_RE.search(text or "")
    if m:
        try:
            out["fps"] = float(m.group(1))
        except ValueError:
            pass
    out["ok"] = out["duration_s"] is not None
    out["tool"] = "ffmpeg"
    return out


def probe_media(path: str, ffprobe: Optional[str], ffmpeg: Optional[str] = None,
                timeout: float = PROBE_TIMEOUT_S) -> dict:
    """Probe one media file: ffprobe JSON first, ``ffmpeg -i`` stderr as fallback. Never raises."""
    res = _empty_probe()
    if ffprobe:
        r = run_cmd([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=r_frame_rate,width,height",
                     "-of", "json", path], timeout=timeout)
        if r.ok:
            try:
                res = parse_ffprobe_json(r.out)
            except (ValueError, json.JSONDecodeError) as e:
                res["error"] = f"ffprobe output unparsable: {e}"
        else:
            res["error"] = (r.err or f"ffprobe rc={r.rc}").strip()[:300]
    if res["duration_s"] is None and ffmpeg:
        r = run_cmd([ffmpeg, "-hide_banner", "-i", path], timeout=timeout)  # exits non-zero by design
        parsed = parse_ffmpeg_stderr(r.err)
        if parsed["duration_s"] is not None:
            res = parsed
        elif not res["error"]:
            res["error"] = (r.err or "ffmpeg -i produced no duration").strip()[-300:]
    res["ok"] = res["duration_s"] is not None
    return res


class ProbeCache:
    """JSONL cache of probe results keyed by ``path|size`` (``data/cache/probe_cache.jsonl``)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and row.get("key"):
                        self.rows[row["key"]] = row
        self._writer = JsonlWriter(self.path)

    @staticmethod
    def key(path: str, size: Optional[int]) -> str:
        return f"{path}|{size if size is not None else '?'}"

    def get(self, key: str) -> Optional[dict]:
        return self.rows.get(key)

    def put(self, key: str, path: str, size: Optional[int], res: dict) -> None:
        row = {"key": key, "path": path, "size": size}
        row.update({k: res.get(k) for k in ("ok", "duration_s", "fps", "width", "height", "error", "tool")})
        self.rows[key] = row
        self._writer.write(row)


def apply_probe(rec: VideoRecord, res: dict, bins: list[float]) -> None:
    """Fill duration/fps/size from a successful probe and refresh the duration stratum."""
    if not res.get("ok") or res.get("duration_s") is None:
        return
    rec.duration_s = float(res["duration_s"])
    if rec.fps is None and res.get("fps"):
        rec.fps = res["fps"]
    if rec.width is None and res.get("width"):
        rec.width = int(res["width"])
    if rec.height is None and res.get("height"):
        rec.height = int(res["height"])
    if rec.strata is None:
        rec.strata = {}
    if rec.strata.get("duration_bin") in (None, "", "unknown"):
        rec.strata["duration_bin"] = duration_bin(rec.duration_s, bins)
    rec.labels["duration_probed"] = True
    note = f"duration probed with {res.get('tool') or 'ffprobe'}"
    if note not in rec.notes:
        rec.notes.append(note)


def probe_records(records: list[VideoRecord], cfg: Config, cache: ProbeCache, workers: int,
                  log: logging.Logger, events: JsonlWriter, dataset: str = "") -> tuple[int, int, int]:
    """Probe ``file`` records with unknown duration; returns (n_from_cache, n_probed_ok, n_failed)."""
    todo = [r for r in records if r.duration_s is None and r.source_kind == "file" and r.source_paths]
    if not todo:
        return 0, 0, 0
    ffprobe = cfg.ffprobe()
    ffmpeg: Optional[str] = None
    try:
        ffmpeg = cfg.ffmpeg()
    except Exception as e:  # pragma: no cover - imageio-ffmpeg is installed
        log.warning("ffmpeg not available for probe fallback: %s", e)
    if not ffprobe and not ffmpeg:
        log.warning("[%s] neither ffprobe nor ffmpeg found; cannot probe %d durations", dataset, len(todo))
        return 0, 0, 0
    if not ffprobe:
        log.warning("[%s] ffprobe not found; parsing `ffmpeg -i` output instead", dataset)
    bins = list(cfg.get("sampling.duration_bins_s", [300, 480, 720]))

    jobs: list[tuple[VideoRecord, str, Optional[int], str]] = []
    n_cached = 0
    for r in todo:
        p = r.source_paths[0]
        try:
            size: Optional[int] = os.path.getsize(p)
        except OSError:
            size = None
        key = ProbeCache.key(p, size)
        hit = cache.get(key)
        if hit and hit.get("ok"):
            apply_probe(r, hit, bins)
            n_cached += 1
        else:
            jobs.append((r, p, size, key))
    log.info("[%s] probing durations: %d records need a duration, %d from cache, %d to probe (workers=%d)",
             dataset, len(todo), n_cached, len(jobs), workers)
    n_ok = n_fail = 0
    if not jobs:
        return n_cached, 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(probe_media, p, ffprobe, ffmpeg): (r, p, size, key) for (r, p, size, key) in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r, p, size, key = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # defensive: probe_media should never raise
                res = _empty_probe()
                res["error"] = f"{type(e).__name__}: {e}"
            cache.put(key, p, size, res)
            if res.get("ok"):
                apply_probe(r, res, bins)
                n_ok += 1
            else:
                n_fail += 1
                r.notes.append(f"probe failed: {res.get('error')}")
                if n_fail <= MAX_ITEM_WARNINGS:
                    log.warning("[%s] probe failed for %s: %s", dataset, r.sample_id, res.get("error"))
                events.write({"event": "probe_failed", "dataset": dataset, "sample_id": r.sample_id,
                              "path": p, "error": res.get("error")})
            if i % 500 == 0 or i == len(jobs):
                log.info("[%s] probe progress %d/%d (%.0f s)", dataset, i, len(jobs), time.time() - t0)
    if n_fail > MAX_ITEM_WARNINGS:
        log.warning("[%s] %d probe failures in total (first %d shown)", dataset, n_fail, MAX_ITEM_WARNINGS)
    return n_cached, n_ok, n_fail


# ------------------------------------------------------------------------------ per-dataset flow
@dataclass
class DatasetResult:
    name: str
    ok: bool = False
    error: Optional[str] = None
    records: list[VideoRecord] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)
    n_raw: int = 0
    n_duplicates: int = 0
    n_probe_cached: int = 0
    n_probed: int = 0
    n_probe_failed: int = 0
    elapsed_s: float = 0.0
    stats: dict = field(default_factory=dict)


def label_histograms(records: list[VideoRecord], max_card: int = MAX_LABEL_CARDINALITY) -> dict[str, dict[str, int]]:
    """Histograms of low-cardinality scalar labels (bool/int/str), e.g. site, experience, flags."""
    counts: dict[str, Counter] = {}
    skip: set[str] = set()
    for r in records:
        for k, v in (r.labels or {}).items():
            if k in skip:
                continue
            if isinstance(v, list):
                vals = [str(x) for x in v if isinstance(x, (str, int, bool))]
            elif isinstance(v, (bool, int, str)):
                vals = [str(v)]
            else:
                continue
            if not vals:
                continue
            c = counts.setdefault(k, Counter())
            for x in vals:
                c[x] += 1
            if len(c) > max_card:
                skip.add(k)
                counts.pop(k, None)
    return {k: dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) for k, c in sorted(counts.items())}


def _iterate_adapter(name: str, cfg: Config, log: logging.Logger, res: DatasetResult) -> Optional[tuple[Any, list[VideoRecord]]]:
    """Import, instantiate and iterate the adapter -> (adapter, records); on failure set res.error, return None."""
    try:
        cls = get_adapter_class(name)
    except Exception as e:
        res.error = f"adapter import failed: {type(e).__name__}: {e}"
        log.error("[%s] %s", name, res.error)
        return None
    try:
        adapter = cls(cfg, log=log)
    except Exception as e:
        res.error = f"adapter init failed: {type(e).__name__}: {e}"
        log.error("[%s] %s", name, res.error, exc_info=True)
        return None
    records: list[VideoRecord] = []
    try:
        for rec in adapter.iter_records():
            records.append(rec)
    except Exception as e:
        res.error = f"adapter raised after {len(records)} records: {type(e).__name__}: {e}"
        log.error("[%s] %s", name, res.error, exc_info=True)
        return None
    return adapter, records


def inventory_dataset(name: str, cfg: Config, log: logging.Logger, events: JsonlWriter,
                      probe: bool, workers: int, cache: Optional[ProbeCache]) -> DatasetResult:
    """Run the whole inventory flow for one dataset (never raises)."""
    res = DatasetResult(name=name)
    t0 = time.time()
    loaded = _iterate_adapter(name, cfg, log, res)
    if loaded is None:
        res.elapsed_s = round(time.time() - t0, 1)
        return res
    adapter, records = loaded
    res.n_raw = len(records)

    # de-duplicate sample ids and sanity-check the dataset name
    seen: set[str] = set()
    kept: list[VideoRecord] = []
    wrong_ds = 0
    for r in records:
        if r.dataset != name:
            wrong_ds += 1
        if r.sample_id in seen:
            res.n_duplicates += 1
            continue
        seen.add(r.sample_id)
        kept.append(r)
    if wrong_ds:
        log.warning("[%s] %d records carry a different dataset name (adapter bug?)", name, wrong_ds)
    if res.n_duplicates:
        log.warning("[%s] %d duplicate sample_ids dropped (first occurrence kept)", name, res.n_duplicates)

    # verify media presence
    checker = PathChecker(log)
    checker.prime(p for r in kept if r.source_kind in ("file", "concat_clips") for p in r.source_paths)
    verified: list[VideoRecord] = []
    for r in kept:
        problems = verify_record(r, checker)
        if problems:
            r.notes.extend(problems)
            res.missing.append({"sample_id": r.sample_id, "video_id": r.video_id, "source_kind": r.source_kind,
                                "first_source_path": r.source_paths[0] if r.source_paths else None,
                                "problems": problems})
            if len(res.missing) <= MAX_ITEM_WARNINGS:
                log.warning("[%s] skipping %s: %s", name, r.sample_id, "; ".join(problems))
        else:
            verified.append(r)
    if len(res.missing) > MAX_ITEM_WARNINGS:
        log.warning("[%s] %d records with missing media in total (first %d shown; see %s.missing.jsonl)",
                    name, len(res.missing), MAX_ITEM_WARNINGS, name)
    log.debug("[%s] path checks: %d scandir, %d stat, %d zip archives opened", name, checker.n_scandir,
              checker.n_stat, checker.n_zip_opened)
    res.records = verified

    if probe and cache is not None:
        res.n_probe_cached, res.n_probed, res.n_probe_failed = probe_records(
            verified, cfg, cache, workers, log, events, dataset=name)

    try:
        stats = adapter.stats(verified)
    except Exception as e:
        log.warning("[%s] adapter.stats failed (%s); using minimal stats", name, e)
        stats = {"dataset": name, "n_records": len(verified)}
    stats.update({
        "ok": True,
        "error": None,
        "n_raw": res.n_raw,
        "n_duplicates": res.n_duplicates,
        "n_missing": len(res.missing),
        "n_duration_unknown": sum(1 for r in verified if r.duration_s is None),
        "n_probe_cached": res.n_probe_cached,
        "n_probed": res.n_probed,
        "n_probe_failed": res.n_probe_failed,
        "label_hist": label_histograms(verified),
        "elapsed_s": round(time.time() - t0, 1),
        "inventoried_at": now_iso(),
        "inventory_file": str(Path(cfg.paths.inventory) / f"{name}.jsonl"),
    })
    res.stats = stats
    res.ok = True
    res.elapsed_s = round(time.time() - t0, 1)
    return res


def write_dataset_outputs(res: DatasetResult, cfg: Config, log: logging.Logger) -> None:
    inv = Path(cfg.paths.inventory)
    n = write_jsonl(inv / f"{res.name}.jsonl", (r.to_dict() for r in res.records))
    miss = inv / f"{res.name}.missing.jsonl"
    if res.missing:
        write_jsonl(miss, res.missing)
    elif miss.exists():
        miss.unlink()
    log.info("[%s] wrote %d records -> %s%s", res.name, n, inv / f"{res.name}.jsonl",
             f" (+{len(res.missing)} missing -> {miss.name})" if res.missing else "")


# ------------------------------------------------------------------------------ summary
def _fmt_hist(hist: dict, limit: int = 12) -> str:
    items = list(hist.items())
    s = ", ".join(f"{k}: {v}" for k, v in items[:limit])
    return s + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def render_summary_md(summary: dict, cfg: Config) -> str:
    lines = ["# Inventory summary", "",
             f"- generated: {now_iso()}",
             f"- datasets root: `{cfg.paths.datasets_root}`",
             f"- inventory dir: `{cfg.paths.inventory}`", "",
             "## Overview", ""]
    rows = []
    for name, s in summary.items():
        status = "ok" if s.get("ok", True) and not s.get("error") else f"ERROR: {s.get('error')}"
        if s.get("last_error"):
            status += f" (last run failed: {s['last_error']})"
        rows.append([name, s.get("title", ""), s.get("n_records", 0), s.get("n_with_duration", "-"),
                     s.get("total_hours", "-"), s.get("mean_duration_s", "-"), s.get("n_missing", "-"),
                     s.get("n_with_segments", "-"),
                     ", ".join(f"{k}: {v}" for k, v in (s.get("source_kinds") or {}).items()) or "-", status])
    lines.append(md_table(["dataset", "title", "records", "with duration", "hours", "mean s", "missing media",
                           "with segments", "source kinds", "status"], rows))
    for name, s in summary.items():
        lines += ["", f"## {name}", ""]
        if s.get("error"):
            lines.append(f"**Error:** {s['error']}")
            continue
        lines.append(f"- records: {s.get('n_records', 0)} (adapter yielded {s.get('n_raw', '-')}, "
                     f"missing media {s.get('n_missing', 0)}, duplicates {s.get('n_duplicates', 0)}); "
                     f"duration unknown: {s.get('n_duration_unknown', '-')}; probed: {s.get('n_probed', 0)} "
                     f"(+{s.get('n_probe_cached', 0)} cached, {s.get('n_probe_failed', 0)} failed)")
        if s.get("procedure_categories"):
            lines.append(f"- procedure categories: {_fmt_hist(s['procedure_categories'])}")
        for k, hist in (s.get("strata") or {}).items():
            lines.append(f"- stratum `{k}`: {_fmt_hist(hist)}")
        for k, hist in (s.get("label_hist") or {}).items():
            lines.append(f"- label `{k}`: {_fmt_hist(hist, 8)}")
        if s.get("notes"):
            lines.append("- notes: " + "; ".join(str(n) for n in s["notes"][:10]))
    return "\n".join(lines) + "\n"


def write_summary(cfg: Config, results: list[DatasetResult], log: logging.Logger) -> dict:
    """Merge this run's statistics into summary.json (keeps entries of datasets not run) and write summary.md."""
    inv = Path(cfg.paths.inventory)
    path = inv / "summary.json"
    summary: dict = {}
    if path.exists():
        try:
            prev = read_json(path)
            if isinstance(prev, dict):
                summary = prev
        except Exception as e:
            log.warning("could not read previous %s (%s); starting fresh", path, e)
    for res in results:
        if res.ok:
            summary[res.name] = res.stats
        elif isinstance(summary.get(res.name), dict) and summary[res.name].get("ok", True) \
                and (inv / f"{res.name}.jsonl").exists():
            # keep the previous good inventory statistics but flag the failure
            summary[res.name]["last_error"] = res.error
            summary[res.name]["last_error_at"] = now_iso()
        else:
            summary[res.name] = {"dataset": res.name, "n_records": 0, "ok": False, "error": res.error,
                                 "inventoried_at": now_iso()}
    ordered = {k: summary[k] for k in DATASETS if k in summary}
    ordered.update({k: v for k, v in summary.items() if k not in ordered})
    write_json(path, ordered)
    (inv / "summary.md").write_text(render_summary_md(ordered, cfg), encoding="utf-8")
    log.info("wrote %s and summary.md", path)
    return ordered


# ------------------------------------------------------------------------------ CLI
def parse_dataset_list(s: Optional[str]) -> list[str]:
    """'a, b,,a' -> ['a', 'b'] (order kept, duplicates removed)."""
    out: list[str] = []
    for part in (s or "").split(","):
        p = part.strip()
        if p and p not in out:
            out.append(p)
    return out


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--datasets", default=None, metavar="a,b",
                   help="comma-separated dataset names (default: every adapter that imports)")
    p.add_argument("--probe-durations", dest="probe_durations", action="store_true",
                   help="run ffprobe (thread pool) for file records whose duration is unknown; cached")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="thread-pool size for probing (default: config `workers`)")


def main(args: argparse.Namespace, cfg: Config) -> int:
    cfg.ensure_dirs()
    log = get_logger(STAGE, cfg)
    started = now_iso()
    t0 = time.time()

    requested = parse_dataset_list(getattr(args, "datasets", None))
    unknown = [d for d in requested if d not in ADAPTER_SPECS]
    if unknown:
        log.error("unknown dataset(s) %s; known: %s", unknown, sorted(ADAPTER_SPECS))
        return 1
    names = requested or [d for d in DATASETS if d in ADAPTER_SPECS]
    probe = bool(getattr(args, "probe_durations", False))
    workers = int(getattr(args, "workers", None) or cfg.workers)

    log.info("inventory start: datasets=%s probe_durations=%s workers=%d", names, probe, workers)
    log.info("config snapshot: %s", json.dumps(redact({
        "datasets_root": str(cfg.paths.datasets_root), "data_dir": str(cfg.paths.data),
        "duration_bins_s": cfg.get("sampling.duration_bins_s"), "ffprobe": cfg.ffprobe() if probe else None,
        "sampling.ophnet.unit": cfg.get("sampling.ophnet.unit"),
    })))
    events = JsonlWriter(Path(cfg.paths.logs) / "inventory_events.jsonl")
    events.write({"event": "start", "stage": STAGE, "datasets": names, "probe_durations": probe})
    cache = ProbeCache(Path(cfg.paths.cache) / "probe_cache.jsonl") if probe else None

    results: list[DatasetResult] = []
    for name in names:
        res = inventory_dataset(name, cfg, log, events, probe, workers, cache)
        results.append(res)
        if res.ok:
            write_dataset_outputs(res, cfg, log)
            hours = res.stats.get("total_hours")
            log.info("[%s] %d records | %d missing media | %d unknown duration | %s h | %.1f s", name,
                     len(res.records), len(res.missing), res.stats.get("n_duration_unknown", 0),
                     f"{hours:.1f}" if isinstance(hours, (int, float)) else "?", res.elapsed_s)
            for k, hist in (res.stats.get("strata") or {}).items():
                log.info("[%s]   stratum %s: %s", name, k, _fmt_hist(hist, 8))
            for k, hist in (res.stats.get("label_hist") or {}).items():
                log.debug("[%s]   label %s: %s", name, k, _fmt_hist(hist, 8))
        events.write({"event": "dataset", "dataset": name, "ok": res.ok, "error": res.error,
                      "n_raw": res.n_raw, "n_records": len(res.records), "n_missing": len(res.missing),
                      "n_duplicates": res.n_duplicates, "n_probe_cached": res.n_probe_cached,
                      "n_probed": res.n_probed, "n_probe_failed": res.n_probe_failed, "elapsed_s": res.elapsed_s})

    write_summary(cfg, results, log)
    ok_names = [r.name for r in results if r.ok]
    failed = {r.name: r.error for r in results if not r.ok}
    summary = {
        "datasets": names, "ok": ok_names, "failed": failed, "probe_durations": probe,
        "n_records": {r.name: len(r.records) for r in results if r.ok},
        "n_missing": {r.name: len(r.missing) for r in results if r.ok},
        "n_probed": sum(r.n_probed for r in results), "n_probe_failed": sum(r.n_probe_failed for r in results),
        "elapsed_s": round(time.time() - t0, 1),
    }
    log.info("inventory done in %.1f s: %s", summary["elapsed_s"],
             ", ".join(f"{k}={v}" for k, v in summary["n_records"].items()) or "no datasets inventoried")
    if failed:
        log.error("failed datasets: %s", failed)
    events.write({"event": "end", "stage": STAGE, **summary})
    record_run(cfg, STAGE, args, summary, started_at=started)
    if not ok_names:
        return 1
    if failed and requested:
        return 1
    return 0
