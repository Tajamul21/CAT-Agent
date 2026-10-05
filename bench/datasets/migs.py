"""MIGS (minimally invasive glaucoma surgery) adapter (DESIGN.md §3.4).

Layout under ``<datasets_root>/MIGS/MIGS/``::

    Task_I_annotation.json       {video id: {annotations: [{label, label_id, segment: [s, e], "segment(frames)": [f0, f1]}],
                                  video_id, clear, "operation type", knife?, "GT incision num"?, subset, fps, duration, frame_count}}
    <id>.mp4                     10 loose videos
    MIGS_video_dataset_<N>.zip   20 deflated zips with the remaining videos; member names like "284 1.mp4"
    Task_II_*.zip, Visualization_Example.zip   instrument-segmentation data / examples (not used here)

File mapping: ``basename without .mp4`` with ``' ' -> '_'`` equals the JSON key (``284 1.mp4`` <-> ``284_1``).
Loose files win over zip members. 21 files have no annotation and are ignored (logged). The JSON key is the
authoritative id: for the 34 ``<n>_1`` ids the embedded ``video_id`` is the bare integer.

Segments keep the dataset's 8 step labels (``Would closure`` is normalised to ``Wound closure``), are sorted by
start and may overlap (Gonioscopy usually contains Goniotomy). ``kind='step'``.
"""
from __future__ import annotations

import logging
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

from bench.config import Config
from bench.datasets.base import DatasetAdapter
from bench.schema import Segment, VideoRecord
from bench.util import make_zip_spec, read_json

CATEGORY = "glaucoma"
ANNOTATION_FILE = "Task_I_annotation.json"
ZIP_GLOB = "MIGS_video_dataset_*.zip"
LABEL_FIXES = {"Would closure": "Wound closure"}

#: operation-type code components -> clinician-readable description
OPERATION_GLOSSARY: dict[str, str] = {
    "GT": "Goniotomy: ab interno incision of the trabecular meshwork under gonioscopic view; the trailing number is "
          "the circumferential extent in degrees (GT120 = 120 degrees, GT240 = 240 degrees, GT360 = full circumference)",
    "PEI": "Phacoemulsification with intraocular lens (IOL) implantation, i.e. combined cataract surgery",
    "GSL": "Goniosynechialysis: mechanical separation of peripheral anterior synechiae from the angle",
    "GATT": "Gonioscopy-assisted transluminal trabeculotomy (suture/microcatheter trabeculotomy, usually 360 degrees)",
    "SPI": "Likely surgical peripheral iridectomy (abbreviation not defined by the dataset authors; interpretation marked 'likely')",
}
OPERATION_SHORT: dict[str, str] = {
    "GT": "goniotomy", "PEI": "phacoemulsification + IOL", "GSL": "goniosynechialysis",
    "GATT": "GATT", "SPI": "surgical peripheral iridectomy (likely)",
}
KNIFE_GLOSSARY: dict[str, str] = {
    "TMH": "Tanito microhook (ab interno trabeculotomy / goniotomy microhook)",
    "KDB": "Kahook Dual Blade (dual-blade goniotomy knife); 'KBD' in the source is a typo",
    "cystotome_needle": "cystotome / bent needle tip used as the goniotomy blade (source text: 破囊针 / 破囊针头 / 针头)",
    "mixed": "more than one instrument recorded (e.g. KDB&TMH, needle&TMH)",
    "unknown": "knife not recorded in the annotation file",
}
#: free-text notes found in the "GT incision num" field (Chinese) -> approximate English
INCISION_NOTE_TRANSLATIONS: dict[str, str] = {
    "看不到GT": "goniotomy not visible in the video",
    "视频不全": "video incomplete",
    "对焦不准": "video out of focus",
    "颞下120忘记录制": "the inferotemporal 120-degree incision was not recorded",
    "后120切开视频不全": "video of the later 120-degree incision is incomplete",
    "给聂-小梁切除术后GT-虹膜萎缩": "goniotomy after previous trabeculectomy; iris atrophy (approximate translation)",
    "视频不全，两段视频只标了1": "video incomplete; two video parts, only one annotated",
    "第二次切开结束时刻因摄像头位置导致记录不清": "end of the second incision unclear because of the camera position",
    "因镜头移动未能观察到第二次切开结束时刻": "end of the second incision not observed because of camera movement",
    "房角分离后即行切开": "incision performed immediately after goniosynechialysis",
    "出血严重、房角分离后即行切开": "severe bleeding; incision performed immediately after goniosynechialysis",
    "房角分离后即行第二次切开": "second incision performed immediately after goniosynechialysis",
}
_DEGREE_RE = re.compile(r"^(GT|GATT)(\d+)?$")


# ------------------------------------------------------------------------------- helpers
def normalise_label(label: str) -> str:
    """Fix known typos in step labels ('Would closure' -> 'Wound closure')."""
    s = str(label).strip()
    return LABEL_FIXES.get(s, s)


def normalise_video_key(name: str) -> str:
    """'284 1.mp4' / '/x/y/284 1.mp4' -> '284_1' (the Task_I_annotation.json key)."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if base.lower().endswith(".mp4"):
        base = base[:-4]
    return base.strip().replace(" ", "_")


def resolve_migs_files(root: str | Path, log: Optional[logging.Logger] = None) -> dict[str, dict[str, Any]]:
    """Map every video id under ``root`` to a file spec (loose ``*.mp4`` first, then ``MIGS_video_dataset_*.zip``).

    Returns ``{id: {"source_kind": "file" | "zip_member", "source_paths": [abs path | "<abs zip>!<member>"],
    "size_bytes": int, "name": original basename, "zip": abs zip path or None, "member": member or None}}``.
    Ids are the normalised basenames (``' ' -> '_'``). Only zip tables of contents are read.
    """
    log = log or logging.getLogger("ophbench.inventory.migs")
    root = Path(root)
    out: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        log.warning("migs: root directory missing: %s", root)
        return out
    for p in sorted(root.glob("*.mp4")):
        out[normalise_video_key(p.name)] = {
            "source_kind": "file", "source_paths": [str(p.resolve())], "size_bytes": p.stat().st_size,
            "name": p.name, "zip": None, "member": None,
        }
    for zp in sorted(root.glob(ZIP_GLOB)):
        try:
            with zipfile.ZipFile(zp) as zf:
                infos = [i for i in zf.infolist() if not i.is_dir()]
        except (zipfile.BadZipFile, OSError) as e:
            log.warning("migs: cannot read zip %s: %s", zp, e)
            continue
        for info in infos:
            if not info.filename.lower().endswith(".mp4"):
                continue
            key = normalise_video_key(info.filename)
            if key in out:
                log.info("migs: %s also found as %s!%s; keeping %s", key, zp.name, info.filename, out[key]["source_paths"][0])
                continue
            out[key] = {
                "source_kind": "zip_member", "source_paths": [make_zip_spec(zp.resolve(), info.filename)],
                "size_bytes": info.file_size, "name": info.filename, "zip": str(zp.resolve()), "member": info.filename,
            }
    return out


def parse_operation_type(code: Any) -> dict[str, Any]:
    """Decode an operation-type code such as ``PEI_GSL_GT120`` or ``PEI+GT``.

    Returns components, goniotomy degrees, booleans per component, an expanded description and the
    glossary entries relevant to this code.
    """
    code_s = str(code or "").strip()
    comps = [c for c in re.split(r"[_+]", code_s) if c]
    degrees: Optional[int] = None
    glossary: dict[str, str] = {}
    expanded: list[str] = []
    bases: set[str] = set()
    for comp in comps:
        m = _DEGREE_RE.match(comp)
        base = m.group(1) if m else comp
        bases.add(base)
        if m and m.group(2):
            deg = int(m.group(2))
            if base == "GT":
                degrees = deg
            expanded.append(f"{OPERATION_SHORT[base]} {deg} degrees")
        else:
            expanded.append(OPERATION_SHORT.get(base, f"{base} (undocumented code)"))
        if base in OPERATION_GLOSSARY:
            glossary[base] = OPERATION_GLOSSARY[base]
        else:
            glossary[base] = "undocumented code (not described by the dataset authors)"
    return {
        "code": code_s or "unknown",
        "components": comps,
        "goniotomy_degrees": degrees,
        "combined_with_phaco": "PEI" in bases,
        "has_goniosynechialysis": "GSL" in bases,
        "has_gatt": "GATT" in bases,
        "has_iridectomy_likely": "SPI" in bases,
        "expanded": "; ".join(expanded) if expanded else "unknown",
        "glossary": glossary,
    }


def normalise_knife(raw: Any) -> str:
    """Collapse knife spellings: TMH | KDB | cystotome_needle | mixed | unknown (unmapped values pass through)."""
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s or s.lower() == "none":
        return "unknown"
    if "&" in s or "+" in s:
        return "mixed"
    if "破囊针" in s or "针头" in s:
        return "cystotome_needle"
    up = s.upper()
    if up == "KBD":
        return "KDB"
    return up if up in ("TMH", "KDB") else s


def _sort_key(video_key: str) -> tuple[int, str]:
    head = video_key.split("_", 1)[0]
    return (int(head) if head.isdigit() else 10 ** 9, video_key)


# ------------------------------------------------------------------------------- adapter
class MigsAdapter(DatasetAdapter):
    """Yield one ``VideoRecord`` per annotated MIGS video (185) with its step timeline."""

    name = "migs"

    def __init__(self, cfg: Config, log: Optional[logging.Logger] = None):
        super().__init__(cfg, log)
        self.ds_root: Path = self.root / "MIGS" / "MIGS"
        self.annotation_path: Path = self.ds_root / ANNOTATION_FILE
        self.unannotated_files: list[str] = []

    # ------------------------------------------------------------------ segments
    @staticmethod
    def build_segments(entry: dict, duration_s: Optional[float]) -> tuple[list[Segment], list[str]]:
        """Step segments sorted by (start, end); overlaps are kept. Returns (segments, notes)."""
        segs: list[Segment] = []
        notes: list[str] = []
        for a in entry.get("annotations", []) or []:
            seg = a.get("segment") or [None, None]
            try:
                start_s, end_s = float(seg[0]), float(seg[1])
            except (TypeError, ValueError, IndexError):
                notes.append(f"annotation without usable segment skipped: {a.get('label')}")
                continue
            if end_s < start_s:
                notes.append(f"segment with end < start swapped: {a.get('label')} {start_s}-{end_s}")
                start_s, end_s = end_s, start_s
            if duration_s is not None and end_s > duration_s + 0.5:
                notes.append(f"segment end {end_s:g}s beyond duration {duration_s:g}s: {a.get('label')}")
            raw_label = str(a.get("label", "")).strip()
            label = normalise_label(raw_label)
            frames = a.get("segment(frames)") or [None, None]
            extra: dict[str, Any] = {"start_frame": frames[0], "end_frame": frames[1]}
            if label != raw_label:
                extra["original_label"] = raw_label
            segs.append(Segment(label=label, label_id=a.get("label_id"), start_s=round(start_s, 3),
                                end_s=round(end_s, 3), kind="step", extra=extra))
        segs.sort(key=lambda s: (s.start_s, s.end_s, s.label))
        return segs, notes

    @staticmethod
    def _has_overlap(segs: list[Segment]) -> bool:
        return any(segs[i + 1].start_s < segs[i].end_s for i in range(len(segs) - 1))

    # ------------------------------------------------------------------ records
    def _make_record(self, key: str, entry: dict, spec: dict[str, Any]) -> VideoRecord:
        duration_s = entry.get("duration")
        duration_s = round(float(duration_s), 3) if duration_s is not None else None
        fps = entry.get("fps")
        fps = round(float(fps), 4) if fps is not None else None
        segments, notes = self.build_segments(entry, duration_s)
        op = parse_operation_type(entry.get("operation type"))
        knife_raw = entry.get("knife")
        knife = normalise_knife(knife_raw)
        incision_raw = entry.get("GT incision num")
        incision_num = incision_raw if isinstance(incision_raw, int) and not isinstance(incision_raw, bool) else None
        incision_note = str(incision_raw).strip() if incision_num is None and incision_raw not in (None, "", "None") else None
        if str(entry.get("video_id")) != key:
            notes.append("multi-part recording: JSON video_id is the bare case number (see labels.json_video_id); "
                         "the JSON key is used as the id")
        if duration_s is None:
            notes.append("duration missing in annotation JSON")
        step_names = list(dict.fromkeys(s.label for s in segments))
        labels: dict[str, Any] = {
            "operation_type": op["code"],
            "operation_components": op["components"],
            "operation_type_expanded": op["expanded"],
            "operation_type_glossary": op["glossary"],
            "goniotomy_degrees": op["goniotomy_degrees"],
            "combined_with_phaco": op["combined_with_phaco"],
            "has_goniosynechialysis": op["has_goniosynechialysis"],
            "has_gatt": op["has_gatt"],
            "has_iridectomy_likely": op["has_iridectomy_likely"],
            "knife": knife_raw,
            "knife_norm": knife,
            "knife_description": KNIFE_GLOSSARY.get(knife, "undocumented knife code"),
            "gt_incision_num": incision_num,
            "gt_incision_note": incision_note,
            "gt_incision_note_en": INCISION_NOTE_TRANSLATIONS.get(incision_note) if incision_note else None,
            "clear": entry.get("clear"),
            "subset": entry.get("subset"),
            "fps": fps,
            "frame_count": entry.get("frame_count"),
            "json_video_id": entry.get("video_id"),
            "n_segments": len(segments),
            "step_names": step_names,
            "n_distinct_steps": len(step_names),
            "has_overlapping_segments": self._has_overlap(segments),
            "file_size_mb": round(spec["size_bytes"] / 1e6, 2) if spec.get("size_bytes") is not None else None,
        }
        strata = {"operation_type": op["code"], "knife": knife, "duration_bin": self.dbin(duration_s)}
        return self.new_record(
            key, source_kind=spec["source_kind"], source_paths=list(spec["source_paths"]), duration_s=duration_s,
            fps=fps, procedure=f"Glaucoma MIGS: {op['code']}", procedure_category=CATEGORY,
            segments=segments, labels=labels, strata=strata, notes=notes,
        )

    def iter_records(self) -> Iterator[VideoRecord]:
        if not self.annotation_path.exists():
            self.log.warning("migs: annotation file missing, skipping dataset: %s", self.annotation_path)
            return
        try:
            data = read_json(self.annotation_path)
        except (OSError, ValueError) as e:
            self.log.warning("migs: cannot parse %s: %s", self.annotation_path, e)
            return
        if not isinstance(data, dict):
            self.log.warning("migs: unexpected annotation structure (%s), skipping", type(data).__name__)
            return
        files = resolve_migs_files(self.ds_root, self.log)
        self.unannotated_files = sorted(set(files) - set(data), key=_sort_key)
        if self.unannotated_files:
            self.log.info("migs: %d video files without annotation ignored: %s", len(self.unannotated_files),
                          ", ".join(self.unannotated_files))
        n_ok = n_skipped = 0
        for key in sorted(data, key=_sort_key):
            entry = data[key]
            spec = files.get(normalise_video_key(key))
            if spec is None or not isinstance(entry, dict):
                self.log.warning("migs: no video file found for annotated id %s, skipped", key)
                n_skipped += 1
                continue
            n_ok += 1
            yield self._make_record(key, entry, spec)
        self.log.info("migs: %d records (%d skipped) from %s", n_ok, n_skipped, self.ds_root)

    # ------------------------------------------------------------------ stats
    def stats(self, records: list[VideoRecord]) -> dict:
        out = super().stats(records)
        out["operation_type_x_knife"] = dict(sorted(Counter(
            f"{r.strata.get('operation_type')}|{r.strata.get('knife')}" for r in records).items()))
        out["step_labels"] = dict(Counter(s.label for r in records for s in r.segments))
        out["n_with_overlapping_segments"] = sum(1 for r in records if r.labels.get("has_overlapping_segments"))
        out["unannotated_files"] = list(self.unannotated_files)
        return out
