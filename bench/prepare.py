"""Prepare stage (DESIGN.md section 6): ``./ophbench prepare``.

For every sampled video (``data/sample/sample_manifest.jsonl``) this stage builds
``data/prepared/batch_<KKK>/<sample_id>/``:

* ``source.mp4``        materialised source (symlink / extracted zip member / concatenated clips)
* ``probe.json``        ProbeInfo (duration, fps, size, codec)
* ``frames/f_###_<t>s.jpg`` timestamp-stamped frames (+ ``frames_index.json`` = list of FrameInfo)
* ``contact_sheet.jpg`` grid of all frames, ``preview.mp4`` (unless ``--no-preview``)
* ``sample.json``       PreparedSample.to_dict() (the contract consumed by generate/export-ui)
* ``prepare.log``       per-sample log with every ffmpeg command

Resumable: samples whose ``sample.json`` exists are skipped unless ``--force``; samples prepared with
``--no-preview`` get only their ``preview.mp4`` added when the stage runs again with previews enabled.
Each attempt appends a row to ``data/prepared/_status.jsonl``.

Selection (DESIGN.md section 13.1): the manifest is put into the canonical benchmark order
(``bench.batches.order_samples`` when available, else the same proportional interleave implemented
here), ``--batch K --batch-size N`` keeps items ``[(K-1)*N, K*N)`` of that order, then ``--ids``,
``--datasets`` and ``--limit`` apply.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

from bench import media
from bench.batches import prepared_dir
from bench.config import Config
from bench.log import JsonlWriter, get_logger, now_iso, record_run
from bench.schema import DATASET_META, FrameInfo, PreparedSample, ProbeInfo, SampleRecord, Segment
from bench.util import fmt_duration, fmt_time, iter_jsonl, read_json, write_json

STAGE = "prepare"
PROGRESS_EVERY = 25
OUTPUT_FILES = ("sample.json", "probe.json", "frames_index.json", "contact_sheet.jpg", "preview.mp4",
                "preview.part.mp4", media.CONCAT_LIST_NAME, media.CONCAT_MAP_NAME, media.SOURCE_NAME,
                "source.part.mp4")


# ------------------------------------------------------------------------------------ CLI
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workers", type=int, default=None, help="thread pool size (default: config `workers`)")
    p.add_argument("--no-preview", action="store_true", help="skip preview.mp4 encoding")
    p.add_argument("--ids", default=None, help="comma-separated sample_ids to prepare")
    p.add_argument("--datasets", default=None, help="comma-separated dataset names to prepare")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N samples of the filtered manifest (stable across runs)")
    p.add_argument("--force", action="store_true", help="redo samples even when sample.json exists")
    p.add_argument("--manifest", default=None,
                   help="sample manifest JSONL (default: <data_dir>/sample/sample_manifest.jsonl)")
    p.add_argument("--batch", type=int, default=None,
                   help="1-based annotation batch to prepare (canonical benchmark order, DESIGN.md 13.1)")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="samples per batch (default: config `batches.size`, 100)")


def _csv_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value]
    else:
        items = str(value).split(",")
    return [s.strip() for s in items if s.strip()]


# ------------------------------------------------------------------------------------ manifest
def load_manifest(path: str | Path, log: Optional[logging.Logger] = None) -> list[SampleRecord]:
    """Read SampleRecords from a JSONL manifest; malformed rows are logged and skipped."""
    records: list[SampleRecord] = []
    for i, row in enumerate(iter_jsonl(path)):
        try:
            records.append(SampleRecord.from_dict(row))
        except Exception as e:  # noqa: BLE001 - one bad row must not kill the stage
            if log:
                log.warning("manifest row %d skipped (%s: %s)", i, type(e).__name__, e)
    return records


def select_records(records: Sequence[SampleRecord], ids: Optional[Sequence[str]] = None,
                   datasets: Optional[Sequence[str]] = None, limit: Optional[int] = None) -> list[SampleRecord]:
    """Filter by sample_id / dataset (manifest order preserved) and keep the first ``limit``."""
    out = list(records)
    if ids:
        wanted = set(ids)
        out = [r for r in out if r.sample_id in wanted]
    if datasets:
        wanted_ds = set(datasets)
        out = [r for r in out if r.dataset in wanted_ds]
    if limit is not None and limit >= 0:
        out = out[:limit]
    return out


def order_samples_fallback(records: Sequence[SampleRecord]) -> list[SampleRecord]:
    """Canonical benchmark order (DESIGN.md 13.1): proportional interleave across datasets.

    Within a dataset the manifest order is kept; item ``i`` of ``n`` gets position ``(i + 0.5) / n`` and the
    global order is ``(position, dataset, sample_id)``, so any prefix (and any batch) holds roughly
    proportional shares of every dataset.
    """
    per_ds: dict[str, list[SampleRecord]] = {}
    for r in records:
        per_ds.setdefault(r.dataset, []).append(r)
    keyed: list[tuple[float, str, str, SampleRecord]] = []
    for ds, items in per_ds.items():
        n = len(items)
        for i, r in enumerate(items):
            keyed.append(((i + 0.5) / n, ds, r.sample_id, r))
    keyed.sort(key=lambda k: (k[0], k[1], k[2]))
    return [k[3] for k in keyed]


def canonical_order(records: Sequence[SampleRecord]) -> list[SampleRecord]:
    """Use ``bench.batches.order_samples`` when that module exists, else :func:`order_samples_fallback`."""
    try:
        from bench.batches import order_samples  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - module written by another stage; optional
        return order_samples_fallback(records)
    try:
        return list(order_samples(list(records)))
    except Exception:  # noqa: BLE001
        return order_samples_fallback(records)


def select_batch(ordered: Sequence[SampleRecord], batch: int, batch_size: int) -> list[SampleRecord]:
    """Items ``[(batch-1)*batch_size, batch*batch_size)`` of the canonical order (1-based batch numbers)."""
    if batch < 1 or batch_size < 1:
        raise ValueError(f"batch and batch size must be >= 1 (got batch={batch}, size={batch_size})")
    try:
        from bench.batches import select_batch as _sb  # type: ignore[import-not-found]
        return list(_sb(list(ordered), batch, batch_size))
    except Exception:  # noqa: BLE001 - fall back to the documented slice
        return list(ordered)[(batch - 1) * batch_size: batch * batch_size]


# ------------------------------------------------------------------------------------ labels / notes
def label_at_t(segments: Sequence[Segment], t: float) -> Optional[str]:
    """Label of the segment(s) covering ``t``; most specific (shortest) first, at most two joined by ' | '."""
    cover = [s for s in segments if float(s.start_s) <= t < float(s.end_s)]
    if not cover:
        cover = [s for s in segments if float(s.start_s) <= t <= float(s.end_s)]
    if not cover:
        return None
    cover.sort(key=lambda s: (s.duration_s, float(s.start_s)))
    labels: list[str] = []
    for s in cover:
        if s.label and s.label not in labels:
            labels.append(s.label)
    return " | ".join(labels[:2]) if labels else None


def realign_concat_segments(rec: SampleRecord, concat_map: Sequence[dict],
                            log: Optional[logging.Logger] = None) -> bool:
    """Snap one-segment-per-clip phase timelines to the probed clip offsets (concat_clips records).

    Only applied when every segment is a phase, there is exactly one per clip, and the record's
    (metadata-derived) starts already agree with the probed offsets to within ~1 s per clip.
    """
    segs = sorted(rec.segments, key=lambda s: float(s.start_s))
    if not segs or len(segs) != len(concat_map) or any(s.kind != "phase" for s in segs):
        return False
    for i, (seg, entry) in enumerate(zip(segs, concat_map)):
        if abs(float(seg.start_s) - float(entry["offset_s"])) > 1.0 + 0.1 * i:
            return False
    for seg, entry in zip(segs, concat_map):
        seg.extra.setdefault("metadata_start_s", round(float(seg.start_s), 3))
        seg.extra.setdefault("metadata_end_s", round(float(seg.end_s), 3))
        seg.start_s = round(float(entry["offset_s"]), 3)
        seg.end_s = round(float(entry["offset_s"]) + float(entry["duration_s"]), 3)
    rec.notes.append("prepare: segment boundaries realigned to probed clip offsets (see concat_map.json)")
    if log:
        log.info("realigned %d segments to probed clip offsets", len(segs))
    return True


def build_timeline_note(rec: SampleRecord, probe: ProbeInfo, concat_map: Optional[Sequence[dict]] = None,
                        realigned: bool = False) -> str:
    """Human/LLM-readable explanation of what the timestamps in this sample refer to."""
    title = DATASET_META.get(rec.dataset, {}).get("title", rec.dataset)
    parts: list[str] = []
    if rec.source_kind == "concat_clips" and concat_map:
        n = len(concat_map)
        shown = ", ".join(f"clip {i} at {fmt_time(e['offset_s'])}" for i, e in enumerate(concat_map[:12]))
        if n > 12:
            shown += f", ... (clip {n - 1} at {fmt_time(concat_map[-1]['offset_s'])})"
        parts.append(
            f"This video is a concatenation of {n} consecutive annotated clips of one {title} case "
            f"({fmt_duration(probe.duration_s)} total); footage between the original clips is absent, so "
            f"transitions between segments can look like jump cuts."
        )
        parts.append(f"All times (segments, frame stamps, preview) refer to the concatenated timeline; "
                     f"clip offsets: {shown} (full table in concat_map.json).")
        if realigned:
            parts.append("Segment boundaries were aligned to the probed clip offsets.")
        if "original_segments" in (rec.labels or {}):
            parts.append("Times in the original source video are kept in labels.original_segments.")
    else:
        parts.append(f"All times refer to the source video as distributed in {title} "
                     f"(probed duration {fmt_duration(probe.duration_s)}).")
        if rec.source_kind == "zip_member":
            parts.append("The file was extracted from the dataset's zip archive without re-encoding.")
    phase_segs = rec.phase_segments
    if phase_segs:
        kinds = ", ".join(sorted({s.kind for s in phase_segs}))
        parts.append(f"{len(phase_segs)} annotated segments ({kinds}) are given in the same timeline.")
        last_end = max(float(s.end_s) for s in phase_segs)
        if last_end > probe.duration_s + 2.0:
            parts.append(f"Note: the annotation timeline extends to {fmt_time(last_end)} although the file "
                         f"probes at {fmt_time(probe.duration_s)}; frames were sampled within the probed duration.")
    else:
        parts.append("No time-localised annotations exist for this video; frames were sampled on a uniform grid.")
    meta_dur = rec.labels.get("metadata_duration_s") if isinstance(rec.labels, dict) else None
    if meta_dur and abs(float(meta_dur) - probe.duration_s) > 2.0:
        parts.append(f"Note: dataset metadata lists {fmt_duration(float(meta_dur))} but the file probes at "
                     f"{fmt_duration(probe.duration_s)}; probed times are used.")
    return " ".join(parts)


def _fill_record_from_probe(rec: SampleRecord, probe: ProbeInfo, log: Optional[logging.Logger] = None) -> None:
    """Fill missing duration/fps/size on the record from the probe; warn on duration disagreement."""
    if rec.duration_s is not None and abs(float(rec.duration_s) - probe.duration_s) > max(2.0, 0.02 * probe.duration_s):
        if log:
            log.warning("metadata duration %.1fs differs from probed %.1fs; using probed value",
                        float(rec.duration_s), probe.duration_s)
        rec.labels["metadata_duration_s"] = round(float(rec.duration_s), 3)
        rec.duration_s = round(probe.duration_s, 3)
    if rec.duration_s is None:
        rec.duration_s = round(probe.duration_s, 3)
    if rec.fps is None and probe.fps:
        rec.fps = round(probe.fps, 3)
    if rec.width is None and probe.width:
        rec.width = probe.width
    if rec.height is None and probe.height:
        rec.height = probe.height


# ------------------------------------------------------------------------------------ per-sample
class _ForwardHandler(logging.Handler):
    """Forward WARNING+ records of a per-sample logger to the stage logger, prefixed with the sample id."""

    def __init__(self, stage_log: logging.Logger, sample_id: str):
        super().__init__(level=logging.WARNING)
        self.stage_log, self.sample_id = stage_log, sample_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.stage_log.log(record.levelno, "[%s] %s", self.sample_id, record.getMessage())
        except Exception:  # pragma: no cover
            pass


def _sample_logger(sample_dir: Path, sample_id: str, stage_log: Optional[logging.Logger]) -> logging.Logger:
    lg = logging.Logger(f"ophbench.prepare.{sample_id}", logging.DEBUG)
    fh = logging.FileHandler(sample_dir / "prepare.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    lg.addHandler(fh)
    if stage_log is not None:
        lg.addHandler(_ForwardHandler(stage_log, sample_id))
    return lg


def _close_logger(lg: logging.Logger) -> None:
    for h in list(lg.handlers):
        lg.removeHandler(h)
        try:
            h.close()
        except Exception:  # pragma: no cover
            pass


def _clean_outputs(sample_dir: Path) -> None:
    for name in OUTPUT_FILES:
        p = sample_dir / name
        if p.is_symlink() or p.exists():
            p.unlink()
    for sub in ("frames", ".tmp"):
        shutil.rmtree(sample_dir / sub, ignore_errors=True)


def _extract_frames(rec: SampleRecord, src: Path, sample_dir: Path, probe: ProbeInfo, media_cfg: dict,
                    cfg: Config, log: logging.Logger) -> tuple[list[FrameInfo], list[Path], list[list[str]]]:
    """Plan, extract (raw to .tmp), overlay into frames/; returns (infos, raw_paths, overlay_lines)."""
    fc = media_cfg.get("frames", {}) if isinstance(media_cfg, dict) else {}
    long_side = int(fc.get("resize_long_side", 768))
    quality = int(fc.get("jpeg_quality", 85))
    segs = rec.phase_segments
    plan = media.plan_frame_times(probe.duration_s, segs, media_cfg)
    if not plan:
        raise media.MediaError(f"empty frame plan for duration {probe.duration_s}")
    log.info("frame plan: %d frames %s", len(plan), dict(Counter(r for _, r in plan)))

    tmp = sample_dir / ".tmp"
    frames_dir = sample_dir / "frames"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    tmp.mkdir(parents=True)
    frames_dir.mkdir(parents=True)

    raw: list[tuple[float, str, Path]] = []
    for i, (t, reason) in enumerate(plan):
        out = tmp / f"raw_{i:03d}.jpg"
        try:
            media.extract_frame(src, t, out, cfg, long_side, log=log)
            raw.append((t, reason, out))
        except media.MediaError as e:
            log.warning("frame %d at t=%.2fs skipped: %s", i, t, e)
    if len(raw) < max(1, len(plan) // 2):
        raise media.MediaError(f"only {len(raw)}/{len(plan)} frames could be extracted")
    if len(raw) < len(plan):
        log.warning("%d of %d planned frames missing", len(plan) - len(raw), len(plan))

    n = len(raw)
    infos: list[FrameInfo] = []
    lines_all: list[list[str]] = []
    for k, (t, reason, raw_path) in enumerate(raw):
        label = label_at_t(segs, t)
        lines = [f"t={fmt_time(t)}  (frame {k + 1}/{n})"] + ([label] if label else [])
        name = f"f_{k:03d}_{t:07.1f}s.jpg"
        media.overlay_text(raw_path, lines, out_path=frames_dir / name, quality=quality)
        infos.append(FrameInfo(idx=k, t_s=round(float(t), 3), file=f"frames/{name}", label_at_t=label, reason=reason))
        lines_all.append(lines)
    return infos, [r[2] for r in raw], lines_all


def prepare_sample(rec: SampleRecord, cfg: Config, media_cfg: Optional[dict] = None, *, no_preview: bool = False,
                   force: bool = False, stage_log: Optional[logging.Logger] = None) -> dict:
    """Run the full per-sample pipeline; never raises (returns a status dict for _status.jsonl)."""
    t0 = time.time()
    media_cfg = media_cfg if media_cfg is not None else cfg.section("media")
    sample_dir = prepared_dir(cfg, rec.sample_id, for_write=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    steps: list[str] = []
    status: dict[str, Any] = {"sample_id": rec.sample_id, "dataset": rec.dataset, "source_kind": rec.source_kind,
                              "ok": False, "steps_done": steps, "error": None, "elapsed_s": None,
                              "n_frames": 0, "preview": False}
    log = _sample_logger(sample_dir, rec.sample_id, stage_log)
    try:
        log.info("=== prepare %s (dataset=%s kind=%s force=%s) ===", rec.sample_id, rec.dataset, rec.source_kind, force)
        if force:
            _clean_outputs(sample_dir)
            log.info("force: previous outputs removed")

        src = media.materialize_source(rec, sample_dir, cfg, log, force=force)
        steps.append("materialize")
        concat_map: Optional[list[dict]] = None
        if rec.source_kind == "concat_clips":
            map_path = sample_dir / media.CONCAT_MAP_NAME
            concat_map = read_json(map_path) if map_path.exists() else None

        probe = media.ffprobe(src, cfg, log)
        write_json(sample_dir / "probe.json", probe.to_dict())
        steps.append("probe")
        log.info("probe: %.2fs %sx%s %.3f fps codec=%s", probe.duration_s, probe.width, probe.height,
                 probe.fps or 0.0, probe.codec)

        realigned = realign_concat_segments(rec, concat_map, log) if concat_map else False
        _fill_record_from_probe(rec, probe, log)

        infos, raw_paths, lines_all = _extract_frames(rec, src, sample_dir, probe, media_cfg, cfg, log)
        write_json(sample_dir / "frames_index.json", [f.to_dict() for f in infos])
        steps.append("frames")
        status["n_frames"] = len(infos)

        cs_cfg = media_cfg.get("contact_sheet", {}) if isinstance(media_cfg, dict) else {}
        fc = media_cfg.get("frames", {}) if isinstance(media_cfg, dict) else {}
        media.build_contact_sheet(raw_paths, sample_dir / "contact_sheet.jpg", cols=int(cs_cfg.get("cols", 6)),
                                  tile_width=int(cs_cfg.get("tile_width", 320)), labels=lines_all,
                                  quality=int(fc.get("jpeg_quality", 85)))
        steps.append("contact_sheet")
        shutil.rmtree(sample_dir / ".tmp", ignore_errors=True)

        preview_name: Optional[str] = None
        pv = media_cfg.get("preview", {}) if isinstance(media_cfg, dict) else {}
        if not no_preview and pv.get("enabled", True):
            media.make_preview(src, sample_dir / "preview.mp4", height=int(pv.get("height", 360)),
                               crf=int(pv.get("crf", 28)), max_duration_s=pv.get("max_duration_s"), cfg=cfg, log=log)
            preview_name = "preview.mp4"
            steps.append("preview")
            status["preview"] = True
        else:
            log.info("preview skipped (%s)", "--no-preview" if no_preview else "disabled in config")

        prepared = PreparedSample(
            record=rec, probe=probe, frames=infos, source_path=media.SOURCE_NAME, frames_dir="frames",
            contact_sheet="contact_sheet.jpg", preview=preview_name, concat_map=concat_map,
            timeline_note=build_timeline_note(rec, probe, concat_map, realigned), prepared_at=now_iso(),
        )
        write_json(sample_dir / "sample.json", prepared.to_dict())
        steps.append("sample_json")
        status.update(ok=True, duration_s=round(probe.duration_s, 3))
        log.info("done in %.1fs: %d frames, preview=%s", time.time() - t0, len(infos), bool(preview_name))
    except Exception as e:  # noqa: BLE001 - per-sample isolation
        status["error"] = f"{type(e).__name__}: {e}"
        log.error("FAILED after %s: %s", steps or "nothing", status["error"])
    finally:
        status["elapsed_s"] = round(time.time() - t0, 2)
        _close_logger(log)
    return status


def add_preview_only(rec: SampleRecord, cfg: Config, media_cfg: Optional[dict] = None, *,
                     stage_log: Optional[logging.Logger] = None) -> dict:
    """Encode ``preview.mp4`` for a sample that was prepared with ``--no-preview`` and update its sample.json.

    Re-running ``prepare`` without ``--no-preview`` therefore only adds the missing previews (no ``--force``,
    no frame re-extraction).  Never raises; returns a status row like :func:`prepare_sample` with
    ``steps_done == ['preview']`` and ``mode == 'preview_only'``.
    """
    t0 = time.time()
    media_cfg = media_cfg if media_cfg is not None else cfg.section("media")
    sample_dir = prepared_dir(cfg, rec.sample_id)
    steps: list[str] = []
    status: dict[str, Any] = {"sample_id": rec.sample_id, "dataset": rec.dataset, "source_kind": rec.source_kind,
                              "ok": False, "steps_done": steps, "error": None, "elapsed_s": None, "n_frames": 0,
                              "preview": False, "mode": "preview_only"}
    log = _sample_logger(sample_dir, rec.sample_id, stage_log)
    try:
        log.info("=== add preview %s (sample.json exists, preview.mp4 missing) ===", rec.sample_id)
        sample_json = sample_dir / "sample.json"
        data = read_json(sample_json)
        status["n_frames"] = len(data.get("frames") or [])
        src = sample_dir / media.SOURCE_NAME
        if not src.exists():  # dangling symlink, cleared extraction cache or deleted concat output
            log.warning("source.mp4 is missing or dangling; re-materialising the source")
            src = media.materialize_source(rec, sample_dir, cfg, log)
        pv = media_cfg.get("preview", {}) if isinstance(media_cfg, dict) else {}
        media.make_preview(src, sample_dir / "preview.mp4", height=int(pv.get("height", 360)), crf=int(pv.get("crf", 28)),
                           max_duration_s=pv.get("max_duration_s"), cfg=cfg, log=log)
        data["preview"] = "preview.mp4"
        write_json(sample_json, data)
        steps.append("preview")
        status.update(ok=True, preview=True, duration_s=(data.get("probe") or {}).get("duration_s"))
        log.info("preview added in %.1fs", time.time() - t0)
    except Exception as e:  # noqa: BLE001 - per-sample isolation
        status["error"] = f"{type(e).__name__}: {e}"
        log.error("FAILED to add preview: %s", status["error"])
    finally:
        status["elapsed_s"] = round(time.time() - t0, 2)
        _close_logger(log)
    return status


# ------------------------------------------------------------------------------------ stage main
def main(args: argparse.Namespace, cfg: Config) -> int:
    started_at = now_iso()
    t0 = time.time()
    cfg.ensure_dirs()
    log = get_logger(STAGE, cfg)
    manifest = Path(getattr(args, "manifest", None) or Path(cfg.paths.sample) / "sample_manifest.jsonl")
    force = bool(getattr(args, "force", False))
    no_preview = bool(getattr(args, "no_preview", False))
    ids = _csv_list(getattr(args, "ids", None))
    datasets = _csv_list(getattr(args, "datasets", None))
    limit = getattr(args, "limit", None)
    workers = int(getattr(args, "workers", None) or cfg.workers)
    media_cfg = cfg.section("media")

    log.info("prepare: manifest=%s data_dir=%s workers=%d force=%s preview=%s", manifest, cfg.paths.data, workers,
             force, not no_preview)
    log.info("media config: %s", media_cfg)
    if not manifest.exists():
        log.error("manifest not found: %s (run `./ophbench sample` first or pass --manifest)", manifest)
        record_run(cfg, STAGE, args, {"error": "manifest not found", "manifest": str(manifest)}, started_at)
        return 1

    records = load_manifest(manifest, log)
    ordered = canonical_order(records)
    batch = getattr(args, "batch", None)
    batch_size = int(getattr(args, "batch_size", None) or cfg.get("batches.size", 100) or 100)
    pool = ordered
    if batch is not None:
        try:
            pool = select_batch(ordered, int(batch), batch_size)
        except ValueError as e:
            log.error("%s", e)
            record_run(cfg, STAGE, args, {"error": str(e)}, started_at)
            return 1
        n_batches = (len(ordered) + batch_size - 1) // batch_size
        log.info("batch %d/%d (size %d): %d samples", batch, n_batches, batch_size, len(pool))
        if not pool:
            log.warning("batch %d is empty (manifest has %d samples -> %d batches of %d)", batch, len(ordered),
                        n_batches, batch_size)
    selected = select_records(pool, ids, datasets, limit)
    log.info("manifest: %d records; selected %d (batch=%s ids=%d datasets=%s limit=%s)", len(records), len(selected),
             batch, len(ids), datasets or "all", limit)

    pv_cfg = media_cfg.get("preview", {}) if isinstance(media_cfg, dict) else {}
    want_preview = not no_preview and bool(pv_cfg.get("enabled", True))
    todo: list[SampleRecord] = []
    preview_todo: list[SampleRecord] = []
    skipped = 0
    for rec in selected:
        sdir = prepared_dir(cfg, rec.sample_id)
        if force or not (sdir / "sample.json").exists():
            todo.append(rec)
        elif want_preview and not (sdir / "preview.mp4").exists():
            preview_todo.append(rec)  # prepared earlier with --no-preview: add the preview only
        else:
            skipped += 1
    log.info("%d already prepared (skipped), %d to do, %d prepared without preview (adding preview.mp4 only)",
             skipped, len(todo), len(preview_todo))

    status_writer = JsonlWriter(Path(cfg.paths.prepared) / "_status.jsonl")
    results: list[dict] = []
    interrupted = False
    n_total = len(todo) + len(preview_todo)
    if n_total:
        ex = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="prepare")
        futs = {ex.submit(prepare_sample, rec, cfg, media_cfg, no_preview=no_preview, force=force,
                          stage_log=log): rec for rec in todo}
        futs.update({ex.submit(add_preview_only, rec, cfg, media_cfg, stage_log=log): rec for rec in preview_todo})
        try:
            for i, fut in enumerate(as_completed(futs), 1):
                rec = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001 - should not happen (prepare_sample catches)
                    res = {"sample_id": rec.sample_id, "dataset": rec.dataset, "source_kind": rec.source_kind,
                           "ok": False, "steps_done": [], "error": f"{type(e).__name__}: {e}", "elapsed_s": None,
                           "n_frames": 0, "preview": False}
                status_writer.write(res)
                results.append(res)
                if res["ok"]:
                    log.info("ok   %-40s frames=%-3d preview=%-5s %.1fs", res["sample_id"], res["n_frames"],
                             res["preview"], res["elapsed_s"] or 0.0)
                else:
                    log.error("FAIL %-40s after %s: %s", res["sample_id"], res.get("steps_done"), res["error"])
                if i % PROGRESS_EVERY == 0 or i == n_total:
                    n_ok = sum(1 for r in results if r["ok"])
                    log.info("progress %d/%d (ok=%d err=%d) elapsed %.0fs", i, n_total, n_ok, i - n_ok,
                             time.time() - t0)
            ex.shutdown(wait=True)
        except KeyboardInterrupt:
            interrupted = True
            log.warning("interrupted: cancelling pending samples (running ffmpeg jobs finish on their own)")
            ex.shutdown(wait=False, cancel_futures=True)

    n_ok = sum(1 for r in results if r["ok"])
    n_err = len(results) - n_ok
    by_ds = Counter(r["dataset"] for r in results if r["ok"])
    errors = Counter((r.get("error") or "").split(":")[0] for r in results if not r["ok"])
    summary = {
        "manifest": str(manifest), "n_manifest": len(records), "n_selected": len(selected), "n_skipped": skipped,
        "n_attempted": len(results), "n_ok": n_ok, "n_error": n_err, "ok_by_dataset": dict(by_ds),
        "n_preview_only": len(preview_todo),
        "error_types": dict(errors), "workers": workers, "preview": not no_preview, "force": force,
        "batch": batch, "batch_size": batch_size if batch is not None else None,
        "interrupted": interrupted, "elapsed_s": round(time.time() - t0, 1),
    }
    log.info("summary: %s", summary)
    record_run(cfg, STAGE, args, summary, started_at)
    if interrupted:
        return 1
    if results and n_ok == 0:
        return 1
    return 0
