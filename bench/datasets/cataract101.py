"""Cataract-101 adapter (DESIGN.md §3.1).

Layout under ``<datasets_root>/cataract-101/``::

    videos.csv        VideoID;Frames;FPS;Surgeon;Experience   101 rows, 25 fps, 720x540, Experience 1=low 2=high
    annotations.csv   VideoID;FrameNo;Phase                   phase *start* frames (1266 rows); a phase runs to the next row
    phases.csv        Phase;Meaning                           10 quasi-standardised phases (ids 1..10)
    videos/case_<VideoID>.mp4

Segment rules (per the dataset README and DESIGN.md):

* rows of a video are sorted by ``FrameNo``; segment ``k`` covers frames ``[start_k, start_{k+1} - 1]``,
  i.e. ``end_s = (next_start - 1) / fps``; the last segment runs to ``Frames`` (= the duration);
* a leading ``Idle/unannotated`` segment covers ``[0, first start)`` when the first start is > 0
  (97 of 101 videos);
* consecutive rows with the same phase (phase 2 occurs twice in almost every surgery, the final
  phase 10 is often split into two sub-annotations) are kept as separate segments, as in the source.

The adapter is a pure metadata reader: no ffmpeg, no video decoding.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Optional

from bench.config import Config
from bench.datasets.base import DatasetAdapter
from bench.schema import Segment, VideoRecord
from bench.util import read_csv_rows

PROCEDURE = "Cataract surgery (phacoemulsification)"
CATEGORY = "cataract"
#: LICENSE.txt in the dataset root is more specific than DATASET_META ("open access for research").
LICENSE = "CC BY-NC 4.0 (LICENSE.txt in the dataset root; cite Schoeffmann et al., MMSys 2018)"
LEAD_LABEL = "Idle/unannotated"
WIDTH, HEIGHT = 720, 540
EXPERIENCE = {1: "low", 2: "high"}
DELIM = ";"


def _to_int(value: Any) -> Optional[int]:
    """Parse an integer cell; None when empty or malformed."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


class Cataract101Adapter(DatasetAdapter):
    """Yield one ``VideoRecord`` per Cataract-101 surgery with its 10-phase timeline."""

    name = "cataract101"

    def __init__(self, cfg: Config, log: Optional[logging.Logger] = None):
        super().__init__(cfg, log)
        self.ds_root: Path = self.root / "cataract-101"
        self.videos_dir: Path = self.ds_root / "videos"

    # ------------------------------------------------------------------ tables
    def _read_tables(self) -> Optional[tuple[list[dict], dict[int, str], dict[str, list[tuple[int, int]]]]]:
        """Read videos.csv, phases.csv and annotations.csv; None (with a WARNING) when a file is missing."""
        paths = {n: self.ds_root / f"{n}.csv" for n in ("videos", "phases", "annotations")}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            self.log.warning("cataract101: metadata missing, skipping dataset: %s", ", ".join(missing))
            return None
        videos = read_csv_rows(paths["videos"], delimiter=DELIM)
        phases: dict[int, str] = {}
        for row in read_csv_rows(paths["phases"], delimiter=DELIM):
            pid = _to_int(row.get("Phase"))
            if pid is not None:
                phases[pid] = str(row.get("Meaning", "")).strip() or f"phase {pid}"
        annotations: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in read_csv_rows(paths["annotations"], delimiter=DELIM):
            frame, phase = _to_int(row.get("FrameNo")), _to_int(row.get("Phase"))
            vid = str(row.get("VideoID", "")).strip()
            if not vid or frame is None or phase is None:
                self.log.warning("cataract101: malformed annotation row skipped: %s", row)
                continue
            annotations[vid].append((frame, phase))
        return videos, phases, annotations

    # ------------------------------------------------------------------ segments
    @staticmethod
    def build_segments(rows: list[tuple[int, int]], frames: int, fps: float, phases: dict[int, str]) -> list[Segment]:
        """Phase segments from (start_frame, phase_id) rows; see module docstring for the rules."""
        rows = sorted(rows)
        segs: list[Segment] = []
        if rows and rows[0][0] > 0:
            lead_end = rows[0][0]
            segs.append(Segment(label=LEAD_LABEL, label_id=None, start_s=0.0, end_s=round(lead_end / fps, 3),
                                kind="phase", extra={"start_frame": 0, "end_frame": lead_end - 1, "synthetic": True}))
        for i, (start, pid) in enumerate(rows):
            end_frame = rows[i + 1][0] - 1 if i + 1 < len(rows) else frames
            end_frame = max(end_frame, start)
            segs.append(Segment(
                label=phases.get(pid, f"phase {pid}"), label_id=pid,
                start_s=round(start / fps, 3), end_s=round(end_frame / fps, 3), kind="phase",
                extra={"start_frame": start, "end_frame": end_frame},
            ))
        return segs

    # ------------------------------------------------------------------ records
    def _make_record(self, row: dict, rows: list[tuple[int, int]], phases: dict[int, str]) -> Optional[VideoRecord]:
        vid_num = _to_int(row.get("VideoID"))
        frames, fps_i = _to_int(row.get("Frames")), _to_int(row.get("FPS"))
        if vid_num is None or not frames or not fps_i:
            self.log.warning("cataract101: malformed videos.csv row skipped: %s", row)
            return None
        video_id = f"case_{vid_num}"
        path = self.videos_dir / f"{video_id}.mp4"
        if not path.exists():
            self.log.warning("cataract101: video file missing, skipping %s (%s)", video_id, path)
            return None
        fps = float(fps_i)
        duration_s = round(frames / fps, 3)
        surgeon = _to_int(row.get("Surgeon"))
        exp_code = _to_int(row.get("Experience"))
        experience = EXPERIENCE.get(exp_code, "low" if exp_code == 1 else "high")
        notes: list[str] = []
        if not rows:
            notes.append("no phase annotations found for this video")
        segments = self.build_segments(rows, frames, fps, phases)
        phase_segs = [s for s in segments if s.label_id is not None]
        counts = Counter(s.label for s in phase_segs)
        labels = {
            "video_number": vid_num,
            "surgeon_id": surgeon,
            "experience": experience,
            "experience_code": exp_code,
            "frames": frames,
            "fps": fps,
            "n_phase_segments": len(phase_segs),
            "n_distinct_phases": len(counts),
            "phase_sequence": [s.label for s in phase_segs],
            "repeated_phases": sorted(k for k, n in counts.items() if n > 1),
            "unannotated_lead_s": round(segments[0].end_s, 3) if segments and segments[0].label == LEAD_LABEL else 0.0,
        }
        strata = {
            "experience": experience,
            "surgeon": str(surgeon) if surgeon is not None else "unknown",
            "duration_bin": self.dbin(duration_s),
        }
        return self.new_record(
            video_id, source_kind="file", source_paths=[str(path)], duration_s=duration_s, fps=fps,
            width=WIDTH, height=HEIGHT, procedure=PROCEDURE, procedure_category=CATEGORY,
            segments=segments, labels=labels, strata=strata, license=LICENSE, notes=notes,
        )

    def iter_records(self) -> Iterator[VideoRecord]:
        tables = self._read_tables()
        if tables is None:
            return
        videos, phases, annotations = tables
        videos.sort(key=lambda r: (_to_int(r.get("VideoID")) is None, _to_int(r.get("VideoID")) or 0))
        n_ok = n_skipped = 0
        for row in videos:
            rec = self._make_record(row, annotations.get(str(row.get("VideoID", "")).strip(), []), phases)
            if rec is None:
                n_skipped += 1
                continue
            n_ok += 1
            yield rec
        self.log.info("cataract101: %d records (%d skipped) from %s", n_ok, n_skipped, self.ds_root)

    # ------------------------------------------------------------------ stats
    def stats(self, records: list[VideoRecord]) -> dict:
        out = super().stats(records)
        out["n_with_leading_unannotated"] = sum(1 for r in records if r.segments and r.segments[0].label == LEAD_LABEL)
        out["phase_segments_per_video"] = dict(sorted(Counter(r.labels.get("n_phase_segments", 0) for r in records).items()))
        out["experience_x_surgeon"] = dict(sorted(Counter(
            f"{r.strata.get('experience')}|surgeon{r.strata.get('surgeon')}" for r in records).items()))
        return out
