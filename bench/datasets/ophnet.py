"""OphNet-2024 adapter (DESIGN.md §3.5).

Inputs under ``<datasets_root>/OphNet2024``:

* ``OphNet2024_loca_challenge_phase.csv`` - one row per phase clip
  (``video_id,start,end,split,surgery_id,phase_id``; CRLF, optional BOM). The i-th row of a case
  (file order, 0-based) is the clip ``OphNet2024_trimmed_phase_extracted/OphNet2024_trimmed_phase/
  <case>/<case>_<i>.mp4``.
* ``OphNet2024_loca_all.csv`` - the finer operation rows (adds ``operation_id``).
* ``OphNet2024_surgery.csv`` - comma-separated surgery ids per case, first = primary.
* ``resources/ophnet_labels.json`` (id -> name/description) and
  ``resources/procedure_taxonomy.yaml: ophnet_surgery_category`` (surgery id -> category).

Two units, chosen with ``cfg.get("sampling.ophnet.unit")``:

* ``case`` (default): one ``concat_clips`` record per case whose segments live in the
  *concatenated* timeline (clip k spans the cumulative ``end-start`` durations of clips 0..k-1).
  Cases longer than ``sampling.ophnet.max_case_duration_s`` keep only the contiguous run of clips
  with the most distinct phases that fits (see :func:`best_window`), recorded in ``notes``.
* ``clip``: one ``file`` record per phase clip.
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from bench.datasets.base import DatasetAdapter
from bench.schema import PROCEDURE_CATEGORIES, Segment, VideoRecord
from bench.util import read_csv_rows

OPHNET_DIR = "OphNet2024"
PHASE_CSV = "OphNet2024_loca_challenge_phase.csv"
OPERATIONS_CSV = "OphNet2024_loca_all.csv"
SURGERY_CSV = "OphNet2024_surgery.csv"
CLIPS_SUBDIR = Path("OphNet2024_trimmed_phase_extracted") / "OphNet2024_trimmed_phase"
LABELS_JSON = "ophnet_labels.json"
TAXONOMY_YAML = "procedure_taxonomy.yaml"

#: The challenge phase file collapses rare phases into id 51 (it appears across 21 surgery types);
#: the label set names id 51 "Instrument Fabrication", which would be misleading here.
OTHER_PHASE_ID = 51
OTHER_PHASE_NAME = "Others (rare phases)"
OTHER_PHASE_DESCRIPTION = ("Bucket used by the challenge annotation files for rare phases outside the "
                           "main phase set (label-set id 51 reads 'Instrument Fabrication').")
#: DESIGN.md notes operation id 106 is also an "Others (rare)" bucket in challenge files; loca_all
#: uses the full 232-operation taxonomy, so the label-set name is kept and a caveat recorded.
OTHER_OPERATION_ID = 106
TIME_EPS = 1e-3
N_PHASES_BINS: tuple[tuple[int, str], ...] = ((3, "1-2"), (6, "3-5"), (10, "6-9"), (15, "10-14"))
CONCAT_NOTE = ("concat_clips: segment times refer to the concatenation of the ordered phase clips; "
               "source-video times are kept in labels.original_segments")


@dataclass
class Clip:
    """One phase clip of a case (times are source-video seconds)."""

    index: int
    file: Path
    start: float
    end: float
    phase_id: int
    surgery_id: int
    split: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, start: float, end: float) -> bool:
        return self.start - TIME_EPS <= start and end <= self.end + TIME_EPS


# ------------------------------------------------------------------------------------ pure helpers
def best_window(durations: list[float], phases: list[Any], max_total: float) -> Optional[tuple[int, int]]:
    """Inclusive (i, j) of the contiguous run with the most distinct phases whose total fits ``max_total``.

    Ties prefer the longer run, then the earlier start. Returns None when not even a single clip fits.
    """
    best_key: Optional[tuple[int, float, int]] = None
    best: Optional[tuple[int, int]] = None
    n = len(durations)
    for i in range(n):
        total = 0.0
        seen: set[Any] = set()
        for j in range(i, n):
            total += durations[j]
            if total > max_total + TIME_EPS:
                break
            seen.add(phases[j])
            key = (len(seen), total, -i)
            if best_key is None or key > best_key:
                best_key, best = key, (i, j)
    return best


def n_phases_bin(n: int) -> str:
    for upper, label in N_PHASES_BINS:
        if n < upper:
            return label
    return f"{N_PHASES_BINS[-1][0]}+"


def parse_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


def parse_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def load_labels(path: Path, log: Any) -> dict[str, dict[str, str]]:
    """``ophnet_labels.json`` -> {surgery, phase, operation, phase_description, operation_description}."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("ophnet: cannot read label names %s: %s (ids will be used as names)", path, e)
        data = {}
    return {k: dict(data.get(k) or {}) for k in ("surgery", "phase", "operation", "phase_description", "operation_description")}


def load_surgery_categories(path: Path, log: Any) -> dict[int, str]:
    """``procedure_taxonomy.yaml: ophnet_surgery_category`` -> {surgery id: category}."""
    try:
        with open(path, encoding="utf-8") as fh:
            tax = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as e:
        log.warning("ophnet: cannot read taxonomy %s: %s (all cases -> other_mixed)", path, e)
        return {}
    out: dict[int, str] = {}
    for category, ids in (tax.get("ophnet_surgery_category") or {}).items():
        if category not in PROCEDURE_CATEGORIES:
            log.warning("ophnet: taxonomy category %r is not a known procedure category", category)
        for sid in ids or []:
            i = parse_int(sid)
            if i is not None:
                out[i] = category
    return out


# ------------------------------------------------------------------------------------ adapter
class OphNetAdapter(DatasetAdapter):
    """743 OphNet-2024 cases (unit=case) or their 14,674 phase clips (unit=clip)."""

    name = "ophnet"

    def __init__(self, cfg: Any, log: Any = None):
        super().__init__(cfg, log)
        self.unit = str(cfg.get("sampling.ophnet.unit", "case") or "case").strip().lower()
        if self.unit not in ("case", "clip"):
            self.log.warning("ophnet: unknown unit %r; using 'case'", self.unit)
            self.unit = "case"
        self.max_case_duration_s = float(cfg.get("sampling.ophnet.max_case_duration_s", 1200) or 0)
        self.names = load_labels(Path(cfg.paths.resources) / LABELS_JSON, self.log)
        self.category_of = load_surgery_categories(Path(cfg.paths.resources) / TAXONOMY_YAML, self.log)

    # ---- paths ------------------------------------------------------------------------------
    @property
    def dataset_root(self) -> Path:
        return self.root / OPHNET_DIR

    @property
    def clips_root(self) -> Path:
        return self.dataset_root / CLIPS_SUBDIR

    # ---- names ------------------------------------------------------------------------------
    def phase_name(self, pid: int) -> str:
        if pid == OTHER_PHASE_ID:
            return OTHER_PHASE_NAME
        return self.names["phase"].get(str(pid), f"Phase {pid}")

    def phase_description(self, pid: int) -> str:
        if pid == OTHER_PHASE_ID:
            return OTHER_PHASE_DESCRIPTION
        return self.names["phase_description"].get(str(pid), "")

    def operation_name(self, oid: int) -> str:
        return self.names["operation"].get(str(oid), f"Operation {oid}")

    def operation_description(self, oid: int) -> str:
        return self.names["operation_description"].get(str(oid), "")

    def surgery_name(self, sid: int) -> str:
        return self.names["surgery"].get(str(sid), f"Surgery {sid}")

    def category(self, sid: Optional[int]) -> str:
        return self.category_of.get(sid, "other_mixed") if sid is not None else "other_mixed"

    # ---- main -------------------------------------------------------------------------------
    def iter_records(self) -> Iterator[VideoRecord]:
        phase_csv = self.dataset_root / PHASE_CSV
        if not phase_csv.exists():
            self.log.warning("ophnet: phase table missing: %s", phase_csv)
            return
        try:
            rows = read_csv_rows(phase_csv)
        except (OSError, csv.Error) as e:
            self.log.warning("ophnet: cannot read %s: %s", phase_csv, e)
            return
        cases = self._group_by_case(rows)
        surgery = self._read_surgery()
        operations = self._read_operations()
        self.log.info("ophnet: %d phase rows in %d cases; unit=%s", len(rows), len(cases), self.unit)
        n = n_trunc = 0
        for case, case_rows in cases.items():
            clips, notes = self._clips_for_case(case, case_rows)
            if not clips:
                continue
            surgery_ids = surgery.get(case) or list(dict.fromkeys(c.surgery_id for c in clips))
            if case not in surgery:
                notes.append("case missing from OphNet2024_surgery.csv; surgery ids taken from the phase rows")
            ops = operations.get(case, [])
            if self.unit == "clip":
                for rec in self._clip_records(case, clips, surgery_ids, ops):
                    n += 1
                    yield rec
            else:
                rec = self._case_record(case, clips, surgery_ids, ops, notes)
                n += 1
                n_trunc += int(bool(rec.labels.get("truncated")))
                yield rec
        self.log.info("ophnet: %d records (%d truncated to max_case_duration_s=%.0f)", n, n_trunc, self.max_case_duration_s)

    # ---- inputs -----------------------------------------------------------------------------
    @staticmethod
    def _group_by_case(rows: list[dict]) -> dict[str, list[dict]]:
        cases: dict[str, list[dict]] = {}
        for row in rows:
            case = (row.get("video_id") or "").strip()
            if case:
                cases.setdefault(case, []).append(row)
        return cases

    def _read_surgery(self) -> dict[str, list[int]]:
        path = self.dataset_root / SURGERY_CSV
        if not path.exists():
            self.log.warning("ophnet: %s missing; surgery ids will come from the phase rows", path)
            return {}
        out: dict[str, list[int]] = {}
        try:
            for row in read_csv_rows(path):
                case = (row.get("video_id") or "").strip()
                ids = [parse_int(x) for x in (row.get("surgery") or "").split(",")]
                ids = [i for i in ids if i is not None]
                if case and ids:
                    out[case] = ids
        except (OSError, csv.Error) as e:
            self.log.warning("ophnet: cannot read %s: %s", path, e)
        return out

    def _read_operations(self) -> dict[str, list[dict[str, Any]]]:
        """loca_all rows mapped to names, grouped by case (source-video times)."""
        path = self.dataset_root / OPERATIONS_CSV
        if not path.exists():
            self.log.warning("ophnet: %s missing; labels.operations will be empty", path)
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        try:
            rows = read_csv_rows(path)
        except (OSError, csv.Error) as e:
            self.log.warning("ophnet: cannot read %s: %s", path, e)
            return {}
        for row in rows:
            case = (row.get("video_id") or "").strip()
            start, end = parse_float(row.get("start")), parse_float(row.get("end"))
            oid, pid = parse_int(row.get("operation_id")), parse_int(row.get("phase_id"))
            if not case or start is None or end is None or oid is None:
                continue
            out.setdefault(case, []).append({
                "operation_id": oid,
                "operation_name": self.operation_name(oid),
                "phase_id": pid,
                "phase_name": self.phase_name(pid) if pid is not None else None,
                "orig_start_s": start,
                "orig_end_s": end,
                "duration_s": round(max(0.0, end - start), 3),
            })
        return out

    def _clips_for_case(self, case: str, rows: list[dict]) -> tuple[list[Clip], list[str]]:
        """Ordered clips of a case whose files exist (one listdir per case); WARNINGs for gaps."""
        case_dir = self.clips_root / case
        if not case_dir.is_dir():
            self.log.warning("ophnet: clip folder missing for %s (%s); case skipped", case, case_dir)
            return [], []
        present = set(os.listdir(case_dir))
        clips: list[Clip] = []
        missing: list[str] = []
        for i, row in enumerate(rows):
            fname = f"{case}_{i}.mp4"
            start, end = parse_float(row.get("start")), parse_float(row.get("end"))
            pid = parse_int(row.get("phase_id"))
            if start is None or end is None or pid is None:
                self.log.warning("ophnet: malformed row %d of %s: %s; clip skipped", i, case, row)
                continue
            if fname not in present:
                missing.append(fname)
                continue
            sid = parse_int(row.get("surgery_id"))
            clips.append(Clip(index=i, file=case_dir / fname, start=start, end=end, phase_id=pid,
                              surgery_id=sid if sid is not None else -1,
                              split=(row.get("split") or "").strip()))
        notes: list[str] = []
        if missing:
            self.log.warning("ophnet: %s is missing %d/%d clip files (e.g. %s)", case, len(missing), len(rows), missing[0])
            notes.append(f"{len(missing)} of {len(rows)} clip files missing on disk and dropped: {', '.join(missing[:5])}")
        return clips, notes

    # ---- case records -----------------------------------------------------------------------
    def _select_window(self, clips: list[Clip], total: float) -> tuple[list[Clip], Optional[str]]:
        """Apply the max_case_duration_s contiguous-window rule."""
        cap = self.max_case_duration_s
        if cap <= 0 or total <= cap + TIME_EPS:
            return clips, None
        window = best_window([c.duration for c in clips], [c.phase_id for c in clips], cap)
        if window is None:
            return clips, (f"case runs {total:.0f} s > max_case_duration_s={cap:.0f} s but no contiguous "
                           f"clip window fits; full case kept")
        i, j = window
        selected = clips[i:j + 1]
        dur = sum(c.duration for c in selected)
        n_ph = len({c.phase_id for c in selected})
        return selected, (f"case runs {total:.0f} s > max_case_duration_s={cap:.0f} s; kept contiguous clips "
                          f"{i}-{j} ({len(selected)} of {len(clips)} clips, {dur:.0f} s, {n_ph} distinct phases)")

    def _case_record(self, case: str, clips: list[Clip], surgery_ids: list[int],
                     ops: list[dict[str, Any]], notes: list[str]) -> VideoRecord:
        total = sum(c.duration for c in clips)
        selected, window_note = self._select_window(clips, total)
        offsets: dict[int, float] = {}
        segments: list[Segment] = []
        offset = 0.0
        for c in selected:
            offsets[c.index] = offset
            segments.append(Segment(
                label=self.phase_name(c.phase_id), label_id=c.phase_id,
                start_s=round(offset, 3), end_s=round(offset + c.duration, 3), kind="phase",
                extra={"clip": c.file.name, "clip_index": c.index, "orig_start_s": c.start, "orig_end_s": c.end,
                       "surgery_id": c.surgery_id},
            ))
            offset += c.duration
        duration = round(offset, 3)

        primary_id = surgery_ids[0]
        primary = self.surgery_name(primary_id)
        category = self.category(primary_id)
        phase_ids = list(dict.fromkeys(c.phase_id for c in selected))
        split = Counter(c.split for c in selected).most_common(1)[0][0]
        max_overlap = max([a.end - b.start for a, b in zip(clips, clips[1:])] + [0.0])
        labels: dict[str, Any] = {
            "case_id": case,
            "split": split,
            "surgery_ids": surgery_ids,
            "surgery_names": [self.surgery_name(s) for s in surgery_ids],
            "primary_surgery": primary,
            "primary_surgery_id": primary_id,
            "n_clips": len(selected),
            "n_phases": len(phase_ids),
            "phase_names": [self.phase_name(p) for p in phase_ids],
            "clips": [self._clip_label(c) for c in selected],
            "original_segments": [{"phase_id": c.phase_id, "phase_name": self.phase_name(c.phase_id),
                                   "start_s": c.start, "end_s": c.end, "surgery_id": c.surgery_id,
                                   "surgery_name": self.surgery_name(c.surgery_id)} for c in selected],
            "operations": [self._operation_in_case(op, selected, offsets) for op in ops],
            "phase_glossary": {self.phase_name(p): self.phase_description(p) for p in phase_ids},
            "operation_glossary": {op["operation_name"]: self.operation_description(op["operation_id"]) for op in ops},
            "full_case_duration_s": round(total, 3),
            "full_case_n_clips": len(clips),
            "full_case_n_phases": len({c.phase_id for c in clips}),
            "truncated": len(selected) != len(clips),
            "clip_window": [selected[0].index, selected[-1].index],
            "clip_overlap_max_s": round(max_overlap, 3),
            "clips_dir": str(self.clips_root / case),
        }
        rec_notes = [CONCAT_NOTE, *notes]
        if window_note:
            rec_notes.append(window_note)
        if max_overlap > TIME_EPS:
            rec_notes.append(f"consecutive source clips overlap by up to {max_overlap:.2f} s in the original video")
        if any(op["operation_id"] == OTHER_OPERATION_ID for op in ops):
            rec_notes.append("operation id 106 may denote 'Others (rare)' in challenge files")
        strata = {
            "primary_surgery": primary,
            "procedure_category": category,
            "duration_bin": self.dbin(duration),
            "n_phases_bin": n_phases_bin(len(phase_ids)),
        }
        return self.new_record(
            case, source_kind="concat_clips", source_paths=[str(c.file) for c in selected], duration_s=duration,
            procedure=primary, procedure_category=category, segments=segments, labels=labels, strata=strata,
            notes=rec_notes,
        )

    def _clip_label(self, c: Clip) -> dict[str, Any]:
        return {"file": c.file.name, "clip_index": c.index, "orig_start_s": c.start, "orig_end_s": c.end,
                "duration_s": round(c.duration, 3), "phase_id": c.phase_id, "phase_name": self.phase_name(c.phase_id),
                "surgery_id": c.surgery_id, "split": c.split}

    @staticmethod
    def _operation_in_case(op: dict[str, Any], selected: list[Clip], offsets: dict[int, float]) -> dict[str, Any]:
        """Add concatenated-timeline times to an operation row (None when outside the kept clips)."""
        out = dict(op)
        host = next((c for c in selected if c.contains(op["orig_start_s"], op["orig_end_s"])), None)
        if host is None:
            out.update(clip_index=None, concat_start_s=None, concat_end_s=None)
        else:
            base = offsets[host.index] - host.start
            out.update(clip_index=host.index, concat_start_s=round(base + op["orig_start_s"], 3),
                       concat_end_s=round(base + op["orig_end_s"], 3))
        return out

    # ---- clip records -----------------------------------------------------------------------
    def _clip_records(self, case: str, clips: list[Clip], surgery_ids: list[int],
                      ops: list[dict[str, Any]]) -> Iterator[VideoRecord]:
        primary_id = surgery_ids[0]
        primary = self.surgery_name(primary_id)
        for c in clips:
            video_id = f"{case}_{c.index}"
            duration = round(c.duration, 3)
            clip_category = self.category(c.surgery_id)
            # a clip-level surgery id that maps to a real category (e.g. trabeculotomy inside a
            # phaco case) describes the clip better than the case primary; otherwise fall back.
            proc_id = c.surgery_id if clip_category != "other_mixed" else primary_id
            segment = Segment(label=self.phase_name(c.phase_id), label_id=c.phase_id, start_s=0.0, end_s=duration,
                              kind="phase", extra={"clip": c.file.name, "orig_start_s": c.start, "orig_end_s": c.end,
                                                   "surgery_id": c.surgery_id})
            clip_ops = []
            for op in ops:
                if c.contains(op["orig_start_s"], op["orig_end_s"]):
                    clip_ops.append({**op, "start_s": round(op["orig_start_s"] - c.start, 3),
                                     "end_s": round(op["orig_end_s"] - c.start, 3)})
            labels: dict[str, Any] = {
                "case_id": case,
                "clip_index": c.index,
                "orig_start_s": c.start,
                "orig_end_s": c.end,
                "phase_id": c.phase_id,
                "phase_name": self.phase_name(c.phase_id),
                "phase_description": self.phase_description(c.phase_id),
                "clip_surgery_id": c.surgery_id,
                "clip_surgery_name": self.surgery_name(c.surgery_id),
                "surgery_ids": surgery_ids,
                "surgery_names": [self.surgery_name(s) for s in surgery_ids],
                "primary_surgery": primary,
                "primary_surgery_id": primary_id,
                "split": c.split,
                "n_phases": 1,
                "case_n_clips": len(clips),
                "operations": clip_ops,
            }
            strata = {
                "primary_surgery": primary,
                "procedure_category": self.category(proc_id),
                "duration_bin": self.dbin(duration),
                "n_phases_bin": n_phases_bin(1),
                "phase": self.phase_name(c.phase_id),
            }
            yield self.new_record(
                video_id, source_kind="file", source_paths=[str(c.file)], duration_s=duration,
                procedure=self.surgery_name(proc_id), procedure_category=self.category(proc_id),
                segments=[segment], labels=labels, strata=strata,
                notes=[f"single phase clip {c.index} of {case} (source-video {c.start:.2f}-{c.end:.2f} s)"],
            )
