"""Cataract-LMM adapters (DESIGN.md §3.3): ``lmm_phase``, ``lmm_skill`` and ``lmm_raw``.

The three datasets share the ``<datasets_root>/Cataract-LMM`` tree (Hugging Face layout):

* ``1_Phase_Recognition`` - 150 complete phacoemulsification videos with 13-phase timelines
  (one CSV per video) stored as members of *uncompressed* zip archives ``videos_0NN.zip``;
* ``4_Skill_Assessment`` - 170 capsulorhexis clips scored by experts (``skill_scores.xlsx``)
  inside deflated zips; 11 clips also exist as loose files under ``<datasets_root>/LMM-samples``
  together with ``manifest.csv`` (Video_ID, Adverse Events, Comment);
* ``5_Raw_Videos`` - 3,000 unannotated complete procedures listed in ``videos_metadata.csv``.

Adapters are pure metadata readers: CSV/XLSX tables plus the ``ZipContentsReport.csv`` tables of
contents. No video is opened. Licence: CC BY-NC-ND 4.0 (recorded in every record's notes).
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Iterator, Optional

from bench.datasets.base import DatasetAdapter
from bench.schema import Segment, VideoRecord
from bench.util import make_zip_spec, read_csv_rows

LMM_DIR = "Cataract-LMM"
LOOSE_SAMPLES_DIR = "LMM-samples"
LICENSE_NOTE = ("Cataract-LMM licence CC BY-NC-ND 4.0: non-commercial research use only; "
                "do not redistribute modified clips.")
PROCEDURE_FULL = "Cataract surgery (phacoemulsification)"
PROCEDURE_SKILL = "Cataract surgery — capsulorhexis phase clip"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

#: Acquisition sites (dataset README). Resolution/fps are *nominal* - a few S2 videos run at 25 fps.
SITES: dict[str, dict[str, Any]] = {
    "S1": {"name": "Farabi Eye Hospital, Tehran", "microscope": "Haag-Streit HS Hi-R NEO 900",
           "nominal_resolution": "720x480", "nominal_fps": 30},
    "S2": {"name": "Noor Eye Hospital, Tehran", "microscope": "ZEISS ARTEVO 800",
           "nominal_resolution": "1920x1080", "nominal_fps": 60},
}

#: CSV phase name -> (P-code from the README, display label). P13 is Idle.
PHASES: dict[str, tuple[int, str]] = {
    "Incision": (1, "Incision"),
    "Viscoelastic": (2, "Viscoelastic"),
    "Capsulorhexis": (3, "Capsulorhexis"),
    "Hydrodissection": (4, "Hydrodissection"),
    "Phacoemulsification": (5, "Phacoemulsification"),
    "IrrigationAspiration": (6, "Irrigation-Aspiration"),
    "CapsulePolishing": (7, "Capsule Polishing"),
    "LensImplantation": (8, "Lens Implantation"),
    "LensPositioning": (9, "Lens Positioning"),
    "ViscoelasticSuction": (10, "Viscoelastic Suction"),
    "AnteriorChamberFlushing": (11, "Anterior Chamber Flushing"),
    "TonifyingAntibiotics": (12, "Tonifying-Antibiotics"),
    "Idle": (13, "Idle"),
}
IDLE = "Idle"

_PHASE_ID_RE = re.compile(r"^PH_(\d{4})_(\d{4})_(S\d)$")
_RAW_ID_RE = re.compile(r"^RV_(\d{4})_(S\d)$")
_SITE_RE = re.compile(r"_(S\d)(?:_|$)")


# ------------------------------------------------------------------------------------ helpers
def site_of(video_id: str) -> str:
    """'PH_0001_2931_S2' / 'SK_0130_S1_P03' / 'RV_0001_S1' -> 'S2' / 'S1' / 'S1' ('unknown' otherwise)."""
    m = _SITE_RE.search(video_id)
    return m.group(1) if m else "unknown"


def site_labels(site: str) -> dict[str, Any]:
    """Label fields describing the acquisition site."""
    info = SITES.get(site, {})
    return {
        "site": site,
        "site_name": info.get("name", "unknown"),
        "microscope": info.get("microscope"),
        "nominal_resolution": info.get("nominal_resolution"),
        "nominal_fps": info.get("nominal_fps"),
    }


def phase_display(csv_name: str) -> tuple[Optional[int], str]:
    """CSV phase name -> (P-code or None, display label)."""
    code, label = PHASES.get(csv_name, (None, csv_name))
    return code, label


def to_float(v: Any) -> Optional[float]:
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def to_int(v: Any) -> Optional[int]:
    f = to_float(v)
    return int(round(f)) if f is not None else None


def text_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def read_zip_report(path: Path, log: Any) -> dict[str, tuple[str, str]]:
    """``ZipContentsReport.csv`` -> ``{member stem: (zip file name, member name)}``.

    The report may carry a UTF-8 BOM and quoted fields (both handled). Missing or malformed
    reports yield an empty mapping after a WARNING, so callers skip rather than crash.
    """
    if not path.exists():
        log.warning("zip contents report missing: %s", path)
        return {}
    out: dict[str, tuple[str, str]] = {}
    try:
        for row in read_csv_rows(path):
            zip_name = (row.get("ZipFileName") or "").strip()
            member = (row.get("InternalFileName") or "").strip()
            if zip_name and member:
                out[Path(member).stem] = (zip_name, member)
    except (OSError, csv.Error) as e:
        log.warning("cannot read zip contents report %s: %s", path, e)
    return out


def tertile_cuts(values: list[float]) -> tuple[float, float]:
    """Lower/upper tertile cut points (values below the first cut are 'low', above the second 'high')."""
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return (float("inf"), float("inf"))
    return (vs[n // 3], vs[(2 * n) // 3])


def tertile_of(value: Optional[float], cuts: tuple[float, float]) -> str:
    if value is None:
        return "unknown"
    if value < cuts[0]:
        return "low"
    if value < cuts[1]:
        return "mid"
    return "high"


# ------------------------------------------------------------------------------------ base
class _LmmAdapter(DatasetAdapter):
    """Shared paths and zip-member resolution for the three Cataract-LMM adapters."""

    @property
    def lmm_root(self) -> Path:
        return self.root / LMM_DIR

    def zip_member_source(self, zip_dir: Path, report: dict[str, tuple[str, str]], stem: str) -> Optional[str]:
        """Absolute ``<zip>!<member>`` spec for ``stem`` or None (after a WARNING) when unresolvable."""
        hit = report.get(stem)
        if not hit:
            self.log.warning("%s: %s is not listed in %s; skipped", self.name, stem, zip_dir / "ZipContentsReport.csv")
            return None
        zip_name, member = hit
        zip_path = zip_dir / zip_name
        if not zip_path.exists():
            self.log.warning("%s: zip archive %s for %s is missing; skipped", self.name, zip_path, stem)
            return None
        return make_zip_spec(zip_path, member)


# ------------------------------------------------------------------------------------ lmm_phase
class LmmPhaseAdapter(_LmmAdapter):
    """150 complete phaco videos with frame-accurate 13-phase timelines (zip members)."""

    name = "lmm_phase"
    IDLE_SHARE_CUTS = (0.2, 0.35)

    @property
    def ann_dir(self) -> Path:
        return self.lmm_root / "1_Phase_Recognition" / "annotations_full_video"

    @property
    def videos_dir(self) -> Path:
        return self.lmm_root / "1_Phase_Recognition" / "videos"

    def iter_records(self) -> Iterator[VideoRecord]:
        if not self.ann_dir.is_dir():
            self.log.warning("%s: annotation folder missing: %s", self.name, self.ann_dir)
            return
        report = read_zip_report(self.videos_dir / "ZipContentsReport.csv", self.log)
        n = 0
        for csv_path in sorted(self.ann_dir.glob("PH_*.csv")):
            rec = self._record_from_csv(csv_path, report)
            if rec is not None:
                n += 1
                yield rec
        self.log.info("%s: %d records", self.name, n)

    def idle_bin(self, share: float) -> str:
        lo, hi = self.IDLE_SHARE_CUTS
        if share < lo:
            return f"<{lo:g}"
        if share <= hi:
            return f"{lo:g}-{hi:g}"
        return f">{hi:g}"

    def _record_from_csv(self, csv_path: Path, report: dict[str, tuple[str, str]]) -> Optional[VideoRecord]:
        video_id = csv_path.stem
        try:
            rows = read_csv_rows(csv_path)
        except (OSError, csv.Error, UnicodeDecodeError) as e:
            self.log.warning("%s: cannot read %s: %s; skipped", self.name, csv_path, e)
            return None
        segments, notes = self._segments(rows, video_id)
        if not segments:
            self.log.warning("%s: no usable segments in %s; skipped", self.name, csv_path)
            return None
        source = self.zip_member_source(self.videos_dir, report, video_id)
        if source is None:
            return None

        duration = max(s.end_s for s in segments)
        idle_s = sum(s.duration_s for s in segments if s.label == IDLE)
        idle_share = idle_s / duration if duration > 0 else 0.0
        phase_names = list(dict.fromkeys(s.label for s in segments))
        site = site_of(video_id)
        m = _PHASE_ID_RE.match(video_id)
        labels: dict[str, Any] = {
            **site_labels(site),
            "subset_index": m.group(1) if m else None,
            "raw_video_id": f"RV_{m.group(2)}_{m.group(3)}" if m else None,
            "idle_share": round(idle_share, 4),
            "idle_seconds": round(idle_s, 3),
            "n_segments": len(segments),
            "n_phases": len([p for p in phase_names if p != IDLE]),
            "phase_names": phase_names,
            "annotation_csv": str(csv_path),
        }
        strata = {"site": site, "duration_bin": self.dbin(duration), "idle_share_bin": self.idle_bin(idle_share)}
        return self.new_record(
            video_id, source_kind="zip_member", source_paths=[source], duration_s=duration,
            fps=self._fps(rows), procedure=PROCEDURE_FULL, procedure_category="cataract",
            segments=segments, labels=labels, strata=strata, notes=[LICENSE_NOTE, *notes],
        )

    def _segments(self, rows: list[dict], video_id: str) -> tuple[list[Segment], list[str]]:
        """Phase segments (sorted by start) plus human-readable notes about repaired rows."""
        segs: list[Segment] = []
        notes: list[str] = []
        for i, row in enumerate(rows, start=2):  # CSV line numbers (header is line 1)
            start, end = to_float(row.get("sec")), to_float(row.get("end_sec"))
            name = (row.get("comment") or "").strip()
            if start is None or end is None or not name:
                self.log.warning("%s: %s line %d is malformed (%s); row skipped", self.name, video_id, i, row)
                notes.append(f"annotation line {i} malformed and skipped")
                continue
            if end < start:
                self.log.warning("%s: %s line %d (%s) ends at %.2f s before its %.2f s start; clamped to zero length",
                                 self.name, video_id, i, name, end, start)
                notes.append(f"annotation line {i} ({name}) ends at {end:.2f} s before its {start:.2f} s start; "
                             f"clamped to zero length, so the true video end is unknown (>= {start:.2f} s)")
                end = start
            code, label = phase_display(name)
            extra: dict[str, Any] = {"raw_name": name}
            f0, f1 = to_int(row.get("frame")), to_int(row.get("end_frame"))
            if f0 is not None and f1 is not None:
                extra.update(start_frame=f0, end_frame=f1)
            segs.append(Segment(label=label, label_id=code, start_s=max(0.0, start), end_s=max(0.0, end),
                                kind="phase", extra=extra))
        segs.sort(key=lambda s: (s.start_s, s.end_s))
        return segs, notes

    @staticmethod
    def _fps(rows: list[dict]) -> Optional[float]:
        """Frame rate implied by the last annotated frame / second pair."""
        best: Optional[tuple[float, float]] = None
        for row in rows:
            end_s, end_f = to_float(row.get("end_sec")), to_float(row.get("end_frame"))
            if end_s and end_f and end_s > 0 and (best is None or end_s > best[0]):
                best = (end_s, end_f)
        return round(best[1] / best[0], 2) if best else None


# ------------------------------------------------------------------------------------ lmm_skill
class LmmSkillAdapter(_LmmAdapter):
    """170 capsulorhexis clips with six 1-5 expert scores, adverse-event flags and comments."""

    name = "lmm_skill"
    SCORE_COLUMNS: dict[str, str] = {
        "Microscope Use": "microscope_use",
        "Instrument Handling": "instrument_handling",
        "Tissue Handling": "tissue_handling",
        "Motion": "motion",
        "Commencement of Flap": "commencement_of_flap",
        "Circular Completion": "circular_completion",
    }

    @property
    def xlsx_path(self) -> Path:
        return self.lmm_root / "4_Skill_Assessment" / "annotation" / "skill_scores.xlsx"

    @property
    def videos_dir(self) -> Path:
        return self.lmm_root / "4_Skill_Assessment" / "videos"

    @property
    def tracking_dir(self) -> Path:
        return self.lmm_root / "3_Object_Tracking" / "annotations"

    @property
    def loose_dir(self) -> Path:
        return self.root / LOOSE_SAMPLES_DIR

    def iter_records(self) -> Iterator[VideoRecord]:
        rows = self._read_scores()
        if not rows:
            return
        report = read_zip_report(self.videos_dir / "ZipContentsReport.csv", self.log)
        manifest = self._read_manifest()
        cuts = tertile_cuts([r["averaged"] for r in rows if r["averaged"] is not None])
        self.log.info("%s: skill tertile cuts on 'Averaged': low < %.3f <= mid < %.3f <= high", self.name, *cuts)
        n = n_loose = 0
        for row in rows:
            rec = self._record(row, report, manifest, cuts)
            if rec is not None:
                n += 1
                n_loose += int(rec.source_kind == "file")
                yield rec
        self.log.info("%s: %d records (%d loose files, %d zip members)", self.name, n, n_loose, n - n_loose)

    # ---- inputs ---------------------------------------------------------------------------
    def _read_scores(self) -> list[dict[str, Any]]:
        """Rows of ``skill_scores.xlsx`` (first sheet) with normalised keys; [] after a WARNING on failure."""
        if not self.xlsx_path.exists():
            self.log.warning("%s: %s missing", self.name, self.xlsx_path)
            return []
        try:
            import openpyxl  # local import keeps the module importable without openpyxl
            wb = openpyxl.load_workbook(self.xlsx_path, read_only=True, data_only=True)
        except Exception as e:  # noqa: BLE001 - any workbook problem is a data problem here
            self.log.warning("%s: cannot open %s: %s", self.name, self.xlsx_path, e)
            return []
        try:
            ws = wb.worksheets[0]
            rows = ws.iter_rows(values_only=True)
            header = next(rows, None)
            if not header:
                self.log.warning("%s: %s is empty", self.name, self.xlsx_path)
                return []
            col = {str(h).strip(): i for i, h in enumerate(header) if h is not None}
            for required in ("Video_ID", "Averaged", "Adverse Events", *self.SCORE_COLUMNS):
                if required not in col:
                    self.log.warning("%s: column %r missing in %s", self.name, required, self.xlsx_path)
            out: list[dict[str, Any]] = []
            for raw in rows:
                vid = text_or_none(_cell(raw, col.get("Video_ID")))
                if not vid:
                    continue
                scores = {key: to_float(_cell(raw, col.get(name))) for name, key in self.SCORE_COLUMNS.items()}
                averaged = to_float(_cell(raw, col.get("Averaged")))
                if averaged is None:
                    present = [v for v in scores.values() if v is not None]
                    averaged = sum(present) / len(present) if present else None
                out.append({
                    "video_id": vid,
                    "phase": text_or_none(_cell(raw, col.get("Phase"))),
                    "scores": scores,
                    "averaged": averaged,
                    "adverse_event": to_int(_cell(raw, col.get("Adverse Events"))),
                    "comment": text_or_none(_cell(raw, col.get("Comment"))),
                })
            return out
        finally:
            wb.close()

    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        """``LMM-samples/manifest.csv`` -> {Video_ID: {adverse_event, comment}} (empty when absent)."""
        path = self.loose_dir / "manifest.csv"
        if not path.exists():
            self.log.info("%s: no loose-sample manifest at %s", self.name, path)
            return {}
        try:
            rows = read_csv_rows(path)
        except (OSError, csv.Error) as e:
            self.log.warning("%s: cannot read %s: %s", self.name, path, e)
            return {}
        return {
            (r.get("Video_ID") or "").strip(): {
                "adverse_event": to_int(r.get("Adverse Events")),
                "comment": text_or_none(r.get("Comment")),
            }
            for r in rows if (r.get("Video_ID") or "").strip()
        }

    # ---- records --------------------------------------------------------------------------
    def _record(self, row: dict[str, Any], report: dict[str, tuple[str, str]],
                manifest: dict[str, dict[str, Any]], cuts: tuple[float, float]) -> Optional[VideoRecord]:
        video_id: str = row["video_id"]
        notes = [LICENSE_NOTE]
        loose = self.loose_dir / f"{video_id}.mp4"
        if loose.exists():
            source_kind, source = "file", str(loose)
            notes.append("loose copy from LMM-samples used instead of the zip member")
        else:
            source_kind = "zip_member"
            source = self.zip_member_source(self.videos_dir, report, video_id)
            if source is None:
                return None

        man = manifest.get(video_id, {})
        site = site_of(video_id)
        parts = video_id.split("_")
        tracking_zip = self._tracking_zip(video_id)
        tertile = tertile_of(row["averaged"], cuts)
        labels: dict[str, Any] = {
            **site_labels(site),
            "subset_index": parts[1] if len(parts) > 1 else None,
            "phase": row["phase"] or "Capsulorhexis",
            "phase_code": parts[3] if len(parts) > 3 else None,
            **row["scores"],
            "skill_scores": dict(row["scores"]),
            "averaged": round(row["averaged"], 4) if row["averaged"] is not None else None,
            "skill_tertile": tertile,
            "adverse_event": row["adverse_event"],
            "comment": row["comment"],
            "adverse_event_comment": man.get("comment") or row["comment"],
            "manifest_adverse_event": man.get("adverse_event"),
            "tracking_annotation_zip": str(tracking_zip) if tracking_zip else None,
            "loose_file": source_kind == "file",
        }
        strata = {
            "skill_tertile": tertile,
            "adverse_event": str(row["adverse_event"]) if row["adverse_event"] is not None else "unknown",
            "site": site,
        }
        return self.new_record(
            video_id, source_kind=source_kind, source_paths=[source], duration_s=None,
            procedure=PROCEDURE_SKILL, procedure_category="cataract", labels=labels, strata=strata, notes=notes,
        )

    def _tracking_zip(self, video_id: str) -> Optional[Path]:
        """``3_Object_Tracking/annotations/TR_<same index>.zip`` when it exists."""
        if not video_id.startswith("SK_"):
            return None
        p = self.tracking_dir / f"TR_{video_id[3:]}.zip"
        return p if p.exists() else None


def _cell(row: Any, idx: Optional[int]) -> Any:
    if idx is None or row is None or idx >= len(row):
        return None
    return row[idx]


# ------------------------------------------------------------------------------------ lmm_raw
class LmmRawAdapter(_LmmAdapter):
    """3,000 unannotated raw procedures from ``videos_metadata.csv``."""

    name = "lmm_raw"

    @property
    def videos_dir(self) -> Path:
        return self.lmm_root / "5_Raw_Videos" / "videos"

    @property
    def metadata_csv(self) -> Path:
        return self.videos_dir / "videos_metadata.csv"

    @property
    def phase_ann_dir(self) -> Path:
        return self.lmm_root / "1_Phase_Recognition" / "annotations_full_video"

    def iter_records(self) -> Iterator[VideoRecord]:
        if not self.metadata_csv.exists():
            self.log.warning("%s: metadata missing: %s", self.name, self.metadata_csv)
            return
        try:
            rows = read_csv_rows(self.metadata_csv)
        except (OSError, csv.Error) as e:
            self.log.warning("%s: cannot read %s: %s", self.name, self.metadata_csv, e)
            return
        files = self._scan_videos()
        phase_subset = self._phase_subset()
        n = n_phase = 0
        for row in rows:
            rec = self._record(row, files, phase_subset)
            if rec is not None:
                n += 1
                n_phase += int(bool(rec.labels.get("in_phase_subset")))
                yield rec
        self.log.info("%s: %d records (%d also in the phase-recognition subset)", self.name, n, n_phase)

    def _scan_videos(self) -> dict[str, str]:
        """One scandir of the videos folder -> {file stem: file name} (prefers .mp4 on stem clashes)."""
        out: dict[str, str] = {}
        if not self.videos_dir.is_dir():
            self.log.warning("%s: videos folder missing: %s", self.name, self.videos_dir)
            return out
        with os.scandir(self.videos_dir) as it:
            for entry in it:
                stem, ext = os.path.splitext(entry.name)
                if ext.lower() not in VIDEO_EXTS:
                    continue
                if stem not in out or ext.lower() == ".mp4":
                    out[stem] = entry.name
        return out

    def _phase_subset(self) -> dict[tuple[str, str], str]:
        """{(raw index RRRR, site Sx): PH video id} from the phase-recognition annotation names."""
        out: dict[tuple[str, str], str] = {}
        if not self.phase_ann_dir.is_dir():
            self.log.warning("%s: phase annotations missing (%s); in_phase_subset will be False", self.name, self.phase_ann_dir)
            return out
        for p in self.phase_ann_dir.glob("PH_*.csv"):
            m = _PHASE_ID_RE.match(p.stem)
            if m:
                out[(m.group(2), m.group(3))] = p.stem
        return out

    def _record(self, row: dict, files: dict[str, str], phase_subset: dict[tuple[str, str], str]) -> Optional[VideoRecord]:
        listed = (row.get("Filename") or "").strip()
        if not listed:
            return None
        stem = Path(listed).stem
        actual = files.get(stem)
        if actual is None:
            self.log.warning("%s: %s listed in metadata but no file under %s; skipped", self.name, listed, self.videos_dir)
            return None
        notes = [LICENSE_NOTE, "unannotated raw procedure (no phase labels)"]
        if actual != listed:
            notes.append(f"metadata lists {listed}; file on disk is {actual}")

        duration = to_float(row.get("Duration (s)"))
        size_mb = to_float(row.get("File Size (MB)"))
        frames = to_int(row.get("Total Frame Count"))
        fps = round(frames / duration, 3) if frames and duration and duration > 0 else None
        site = site_of(stem)
        m = _RAW_ID_RE.match(stem)
        key = (m.group(1), m.group(2)) if m else None
        phase_video_id = phase_subset.get(key) if key else None
        labels: dict[str, Any] = {
            **site_labels(site),
            "raw_index": m.group(1) if m else None,
            "file_size_mb": round(size_mb, 3) if size_mb is not None else None,
            "frame_count": frames,
            "in_phase_subset": phase_video_id is not None,
            "phase_video_id": phase_video_id,
            "metadata_filename": listed,
            "file_name": actual,
        }
        strata = {"site": site, "duration_bin": self.dbin(duration)}
        return self.new_record(
            stem, source_kind="file", source_paths=[str(self.videos_dir / actual)],
            duration_s=round(duration, 3) if duration is not None else None, fps=fps,
            procedure=PROCEDURE_FULL, procedure_category="cataract", labels=labels, strata=strata, notes=notes,
        )
