"""Cataract-1K adapter (DESIGN.md §3.2).

Layout under ``<datasets_root>/cataract-1k/``::

    cat-1k/case_<id>.mp4                       1000 videos (case_2000 .. case_2999; 640x360, ~60 fps on disk)
    cat-1k/SYNAPSE_METADATA_MANIFEST.tsv       Synapse download manifest (path/name/synapse id per video) - ignored
    cataract-1k_annotations/videos.csv         VideoID,Frames,FPS     annotated videos; FPS == 1 so Frames == seconds
    cataract-1k_annotations/annotations.csv    VideoID,FrameNo,Phase  1 fps: FrameNo == seconds; a row starts a label
                                               that lasts until the next row; the final row sits at FrameNo == Frames
                                               and is an end marker (zero length -> dropped)
    cataract-1k_annotations/phases.csv         Phase,Meaning          0 Idle .. 13 Suture; 14/15/16 are usage flags

Rules:

* one record per ``*.mp4`` found in ``cat-1k`` (listed once with ``os.scandir``); annotated ids whose
  video is not on disk are skipped with a WARNING (56 ids of the 303 in ``videos.csv`` in this copy);
* phases 0-13 become ``kind='phase'`` segments, rows 14/15/16 become ``kind='flag'`` segments *and*
  are collected (names) in ``labels.flags``;
* unannotated videos have ``duration_s=None`` (``inventory --probe-durations`` fills it) and are
  binned by file size instead: ``strata.size_bin`` uses quartiles of the file sizes over the whole
  dataset (small / medium / large / xlarge). ``size_bin`` is set for every record so the sampler can
  always fall back to it.
"""
from __future__ import annotations

import logging
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Optional

from bench.config import Config
from bench.datasets.base import DatasetAdapter
from bench.schema import Segment, VideoRecord
from bench.util import read_csv_rows

PROCEDURE = "Cataract surgery (phacoemulsification)"
CATEGORY = "cataract"
WIDTH, HEIGHT = 640, 360
#: fallback names when phases.csv lacks a row; phases.csv wins when present
FLAG_NAMES = {14: "Trypan Blue Injection Used", 15: "Iris Hooks Used", 16: "Malyugin Ring Used"}
IDLE_ID = 0
NOT_CATARACT_ID = 10
SUTURE_ID = 13
SIZE_BIN_LABELS = ("small", "medium", "large", "xlarge")


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def size_quartile_cuts(sizes: list[int]) -> Optional[list[float]]:
    """Three quartile cut points (Q1, Q2, Q3) over the given byte sizes; None when too few values."""
    if len(sizes) < 4:
        return None
    try:
        return [float(q) for q in statistics.quantiles(sizes, n=4, method="exclusive")]
    except statistics.StatisticsError:
        return None


def size_bin(size_bytes: Optional[int], cuts: Optional[list[float]]) -> str:
    """Map a file size onto small/medium/large/xlarge using the quartile cuts (None -> 'unknown')."""
    if size_bytes is None or not cuts:
        return "unknown"
    for label, cut in zip(SIZE_BIN_LABELS, cuts):
        if size_bytes < cut:
            return label
    return SIZE_BIN_LABELS[-1]


class Cataract1kAdapter(DatasetAdapter):
    """Yield one ``VideoRecord`` per Cataract-1K video (1000), 247 of them with 1-fps phase timelines."""

    name = "cataract1k"

    def __init__(self, cfg: Config, log: Optional[logging.Logger] = None):
        super().__init__(cfg, log)
        self.ds_root: Path = self.root / "cataract-1k"
        self.videos_dir: Path = self.ds_root / "cat-1k"
        self.ann_dir: Path = self.ds_root / "cataract-1k_annotations"
        self.size_cuts_mb: Optional[list[float]] = None
        self.n_annotated_without_video: int = 0

    # ------------------------------------------------------------------ files
    def scan_videos(self) -> dict[int, tuple[Path, int]]:
        """Single ``os.scandir`` pass over cat-1k: video number -> (path, size_bytes). Non-mp4 entries are logged."""
        found: dict[int, tuple[Path, int]] = {}
        if not self.videos_dir.is_dir():
            self.log.warning("cataract1k: video directory missing: %s", self.videos_dir)
            return found
        for entry in os.scandir(self.videos_dir):
            if not entry.name.lower().endswith(".mp4"):
                self.log.info("cataract1k: ignoring non-mp4 entry %s", entry.name)
                continue
            stem = entry.name[:-4]
            num = _to_int(stem[5:]) if stem.startswith("case_") else None
            if num is None:
                self.log.warning("cataract1k: unexpected video name %s (want case_<id>.mp4), skipped", entry.name)
                continue
            try:
                size = entry.stat().st_size
            except OSError as e:  # pragma: no cover - NFS hiccup
                self.log.warning("cataract1k: cannot stat %s: %s", entry.path, e)
                size = 0
            found[num] = (Path(entry.path), size)
        return found

    # ------------------------------------------------------------------ tables
    def _read_annotations(self) -> tuple[dict[int, tuple[int, float]], dict[int, list[tuple[int, int]]], dict[int, str]]:
        """(videos: id -> (Frames, FPS), rows: id -> [(FrameNo, Phase)], phase names). Empty dicts when missing."""
        videos: dict[int, tuple[int, float]] = {}
        rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
        phases: dict[int, str] = dict(FLAG_NAMES)
        paths = {n: self.ann_dir / f"{n}.csv" for n in ("videos", "annotations", "phases")}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            self.log.warning("cataract1k: annotation files missing (all videos treated as unannotated): %s", ", ".join(missing))
            return videos, rows, phases
        for row in read_csv_rows(paths["phases"]):
            pid = _to_int(row.get("Phase"))
            if pid is not None:
                phases[pid] = str(row.get("Meaning", "")).strip() or phases.get(pid, f"phase {pid}")
        for row in read_csv_rows(paths["videos"]):
            vid, frames, fps = _to_int(row.get("VideoID")), _to_int(row.get("Frames")), _to_int(row.get("FPS"))
            if vid is None or not frames or not fps:
                self.log.warning("cataract1k: malformed videos.csv row skipped: %s", row)
                continue
            videos[vid] = (frames, float(fps))
        for row in read_csv_rows(paths["annotations"]):
            vid, frame, phase = _to_int(row.get("VideoID")), _to_int(row.get("FrameNo")), _to_int(row.get("Phase"))
            if vid is None or frame is None or phase is None:
                self.log.warning("cataract1k: malformed annotation row skipped: %s", row)
                continue
            rows[vid].append((frame, phase))
        return videos, rows, phases

    # ------------------------------------------------------------------ segments
    @staticmethod
    def build_segments(rows: list[tuple[int, int]], duration_s: float, fps: float,
                       phases: dict[int, str]) -> tuple[list[Segment], list[str]]:
        """Segments from 1-fps rows: each row lasts until the next row, the last one until ``duration_s``.

        Zero-length rows are dropped: the final row of every annotated video sits at ``FrameNo == Frames``
        and only marks the end. Phase ids 14/15/16 become ``kind='flag'``. Returns (segments, notes).
        """
        rows = sorted(rows)
        last_frame = int(round(duration_s * fps))
        segs: list[Segment] = []
        notes: list[str] = []
        unexpected_zero = 0
        for i, (frame, pid) in enumerate(rows):
            if frame > last_frame:
                notes.append(f"annotation row at frame {frame} beyond the video end ({last_frame}) dropped")
                continue
            end_frame = min(rows[i + 1][0] if i + 1 < len(rows) else last_frame, last_frame)
            if end_frame <= frame:
                unexpected_zero += frame < last_frame  # the end marker at last_frame is expected
                continue
            kind = "flag" if pid in FLAG_NAMES else "phase"
            segs.append(Segment(label=phases.get(pid, f"phase {pid}"), label_id=pid, start_s=round(frame / fps, 3),
                                end_s=round(end_frame / fps, 3), kind=kind,
                                extra={"start_frame": frame, "end_frame": end_frame}))
        if unexpected_zero:
            notes.append(f"{unexpected_zero} zero-length annotation rows dropped")
        if rows and not segs:
            notes.append("annotation contains only the end-marker row; no phase segments")
        return segs, notes

    # ------------------------------------------------------------------ records
    def _make_record(self, num: int, path: Path, size: int, ann: Optional[tuple[int, float]],
                     rows: list[tuple[int, int]], phases: dict[int, str], cuts: Optional[list[float]]) -> VideoRecord:
        video_id = f"case_{num}"
        size_mb = round(size / 1e6, 2)
        labels: dict[str, Any] = {
            "video_number": num, "annotated": False, "flags": [], "n_phase_segments": 0,
            "has_not_cataract": False, "has_suture": False, "file_size_mb": size_mb,
        }
        strata: dict[str, Any] = {"annotated": False, "rare_flag": False, "duration_bin": "unknown",
                                  "size_bin": size_bin(size, cuts)}
        duration_s: Optional[float] = None
        segments: list[Segment] = []
        notes: list[str] = []
        if ann is not None:
            frames, ann_fps = ann
            duration_s = round(frames / ann_fps, 3)
            segments, notes = self.build_segments(rows, duration_s, ann_fps, phases)
            phase_segs = [s for s in segments if s.kind == "phase"]
            flag_segs = [s for s in segments if s.kind == "flag"]
            flags = sorted({s.label for s in flag_segs})
            ids = {s.label_id for s in phase_segs}
            idle_s = sum(s.duration_s for s in phase_segs if s.label_id == IDLE_ID)
            labels.update(
                annotated=True, flags=flags, n_phase_segments=len(phase_segs), n_flag_segments=len(flag_segs),
                has_not_cataract=NOT_CATARACT_ID in ids, has_suture=SUTURE_ID in ids,
                frames=frames, annotation_fps=ann_fps,
                phase_sequence=[s.label for s in phase_segs],
                n_distinct_phases=len(ids),
                idle_share=round(idle_s / duration_s, 3) if duration_s else None,
            )
            strata.update(
                annotated=True,
                rare_flag=bool(flags) or labels["has_suture"] or labels["has_not_cataract"],
                duration_bin=self.dbin(duration_s),
            )
        return self.new_record(
            video_id, source_kind="file", source_paths=[str(path)], duration_s=duration_s, fps=None,
            width=WIDTH, height=HEIGHT, procedure=PROCEDURE, procedure_category=CATEGORY,
            segments=segments, labels=labels, strata=strata, notes=notes,
        )

    def iter_records(self) -> Iterator[VideoRecord]:
        files = self.scan_videos()
        if not files:
            return
        videos, rows, phases = self._read_annotations()
        missing = sorted(v for v in videos if v not in files)
        self.n_annotated_without_video = len(missing)
        if missing:
            self.log.warning("cataract1k: %d annotated ids have no video in %s (skipped): %s%s", len(missing),
                             self.videos_dir, ", ".join(f"case_{v}" for v in missing[:8]), " ..." if len(missing) > 8 else "")
        cuts = size_quartile_cuts([s for _, s in files.values()])
        self.size_cuts_mb = [round(c / 1e6, 2) for c in cuts] if cuts else None
        self.log.info("cataract1k: %d videos, %d annotated; size quartile cuts (MB): %s",
                      len(files), sum(1 for v in files if v in videos), self.size_cuts_mb)
        for num in sorted(files):
            path, size = files[num]
            yield self._make_record(num, path, size, videos.get(num), rows.get(num, []), phases, cuts)

    # ------------------------------------------------------------------ stats
    def stats(self, records: list[VideoRecord]) -> dict:
        out = super().stats(records)
        out["n_annotated"] = sum(1 for r in records if r.labels.get("annotated"))
        out["n_annotated_without_video"] = self.n_annotated_without_video
        out["n_rare_flag"] = sum(1 for r in records if r.strata.get("rare_flag"))
        out["flags"] = dict(Counter(f for r in records for f in r.labels.get("flags", [])))
        out["size_bin_cuts_mb"] = self.size_cuts_mb
        return out
