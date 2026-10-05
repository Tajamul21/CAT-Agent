"""Export stage (DESIGN.md §9): build the static bundle read by the clinician annotation UI.

Inputs
    data/sample/sample_manifest.jsonl      SampleRecord rows (export order)
    data/prepared/batch_<KKK>/<sample_id>/sample.json  PreparedSample (probe, frames, contact sheet, preview)
    data/qa/qa_validated.jsonl             QASet rows + ``validation`` (falls back to data/qa/<id>.json
                                           when validation has not run; stale rows are detected via
                                           ``provenance.generated_at``)

Outputs (all media URLs are relative to the ui/ folder: ``media/<sample_id>/...``)
    ui/data/meta.json                      {generated_at, ui_version, n_samples, datasets, select_min,
                                            select_max, annotators, dataset_meta, ...}
    ui/data/index.json                     [{sample_id, dataset, dataset_title, procedure, procedure_category,
                                             duration_s, n_questions, thumb, preview, assigned_to, has_issues}]
    ui/data/samples/<sample_id>.json       full sample document (video, metadata, segments, frames, questions)
    ui/data/assignments.json               {annotators, by_annotator, overlap}
    ui/media/<sample_id>/contact_sheet.jpg, preview.mp4   (symlink | copy | none)

Every sample with a prepared folder and a QA set is exported; items that failed validation are
exported too, marked ``has_issues`` with the codes in each question's ``validation_issues``.
Samples without QA (or without a prepared folder) are skipped (one summary WARNING).

Batches (DESIGN.md section 13): ``--batch K`` exports only batch K of the canonical benchmark order and is
*additive* - index.json / assignments.json / meta.json keep the previously exported batches; every entry
and sample document carries ``"batch": K`` and ``video.local_file`` (``<sample_id>.mp4``).  Unless
``--no-package`` an offline package ``ui_packages/batch_<K:03d>/`` with one video per sample, MANIFEST.csv
and README.txt is written for sharing with the clinicians (``--package-kind preview|source``).

The UI output root is ``cfg.paths.ui`` (``<repo>/ui``) unless the environment variable
``OPHBENCH_UI_DIR`` is set (used by the smoke test and the unit tests so they never clobber ui/).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from bench.batches import (PACKAGE_KINDS, assign_batches, batch_dir_name, batch_size, n_batches, order_samples,
                           package_kind, packages_dir, ranks, select_batch)
from bench.batches import prepared_dir as resolve_prepared_dir
from bench.config import Config
from bench.log import get_logger, now_iso, record_run, redact
from bench.schema import DATASET_META, UI_SAMPLE_VERSION, PreparedSample, ProbeInfo, QASet, SampleRecord
from bench.util import fmt_duration, human_size, iter_jsonl, read_json, write_csv, write_json

STAGE = "export_ui"
#: symlink / copy place contact sheet + preview; ``sheets`` copies only the contact sheet (videos are shared
#: offline, DESIGN.md 13.2 - the default with --batch); ``none`` places nothing.
MEDIA_MODES = ("symlink", "copy", "none", "sheets")
PACKAGE_MANIFEST_FIELDS = ["sample_id", "dataset", "procedure", "duration_s", "file", "size_mb", "batch", "kind"]

CATEGORY_LABELS = {
    "cataract": "Cataract", "glaucoma": "Glaucoma", "cornea": "Cornea / ocular surface", "retina": "Retina",
    "oculoplastics_strabismus": "Oculoplastics / strabismus", "refractive": "Refractive", "other_mixed": "Other / mixed",
}

# label-dict keys -> clinician-friendly names (unknown keys are humanised automatically).
# Tuned to the keys the adapters actually emit (checked against a real inventory run).
LABEL_NAMES: dict[str, str] = {
    # provenance / centre
    "site": "Site code", "site_name": "Centre", "microscope": "Microscope", "surgeon_id": "Surgeon", "surgeon": "Surgeon",
    "experience": "Surgeon experience",
    # phase annotations (Cataract-101 / Cataract-1K / Cataract-LMM / OphNet)
    "annotated": "Phase annotations available", "n_phase_segments": "Annotated phase segments",
    "n_segments": "Annotated segments", "n_distinct_phases": "Distinct phases", "n_phases": "Distinct phases",
    "phase_sequence": "Phase sequence", "phase_names": "Phases present", "repeated_phases": "Repeated phases",
    "unannotated_lead_s": "Unannotated lead-in", "idle_share": "Idle share of video", "idle_seconds": "Idle time",
    "flags": "Special techniques / flags", "has_not_cataract": "Non-cataract phase present", "has_suture": "Suturing present",
    # Cataract-LMM skill assessment
    "phase": "Phase shown", "microscope_use": "Skill: microscope use", "instrument_handling": "Skill: instrument handling",
    "tissue_handling": "Skill: tissue handling", "motion": "Skill: motion", "commencement_of_flap": "Skill: commencement of flap",
    "circular_completion": "Skill: circular completion", "averaged": "Skill score (average of 6)",
    "skill_tertile": "Skill tertile (within dataset)", "scores": "Skill scores (1-5)",
    "adverse_event": "Adverse event", "comment": "Expert comment", "adverse_event_comment": "Adverse event comment",
    # MIGS
    "operation_type": "Operation type (code)", "operation_type_expanded": "Operation type",
    "operation_type_glossary": "Operation code glossary", "goniotomy_degrees": "Goniotomy extent (degrees)",
    "combined_with_phaco": "Combined with phacoemulsification", "has_goniosynechialysis": "Goniosynechialysis performed",
    "has_gatt": "GATT performed", "has_iridectomy_likely": "Surgical iridectomy (likely)", "knife": "Knife (code)",
    "knife_description": "Knife", "gt_incision_num": "Incision count (ground truth)", "gt_incision_note_en": "Incision note",
    "clear": "Clear view", "step_names": "Steps present", "n_distinct_steps": "Distinct steps",
    "has_overlapping_segments": "Overlapping step labels",
    # OphNet
    "surgery_names": "Surgery type(s)", "primary_surgery": "Primary surgery", "n_clips": "Phase clips in case",
    "operations": "Fine-grained operations", "full_case_duration_s": "Full case duration",
    "full_case_n_clips": "Full case: phase clips", "full_case_n_phases": "Full case: distinct phases",
    "truncated": "Case truncated to fit the duration cap",
    # Ophora
    "instruction": "Narration / caption", "source_video_id": "Source video (YouTube id)", "clip_index": "Clip index",
    "category_keywords": "Matched procedure keywords",
}
# internal / redundant keys never shown to clinicians
DROP_KEYS = frozenset({
    # paths, ids, file facts
    "tracking_annotation_zip", "annotation_csv", "clips_dir", "metadata_filename", "file_name", "loose_file",
    "video_number", "case_id", "json_video_id", "raw_video_id", "raw_index", "subset_index", "phase_video_id",
    "primary_surgery_id", "surgery_ids", "phase_code", "experience_code", "label_ids", "phase_ids",
    "zip", "zip_path", "member", "path", "paths", "file", "files",
    # redundant with the video block or other rows
    "frames", "frame_count", "fps", "annotation_fps", "nominal_fps", "nominal_resolution", "nominal_duration_s",
    "file_size_mb", "n_flag_segments", "skill_scores", "manifest_adverse_event", "operation_components", "knife_norm",
    "knife_description_short", "gt_incision_note", "clip_window", "clip_overlap_max_s",
    # structured / prompt-only material (the UI shows segments and frames instead)
    "clips", "original_segments", "phase_glossary", "operation_glossary", "split", "subset", "in_phase_subset",
    "in_filtered_28k", "instruction_words",
})
MAX_LIST_ITEMS = 20
MAX_TEXT_CHARS = 600

_PATH_RE = re.compile(r"(^/)|(^[A-Za-z]:\\)|(\.(mp4|mkv|avi|mov|zip|csv|json|jsonl|xlsx|jpg|jpeg|png|txt)$)", re.I)


# ------------------------------------------------------------------------------------ paths
def ui_paths(cfg: Config) -> SimpleNamespace:
    """UI output folders; honours ``OPHBENCH_UI_DIR`` (tests / smoke runs) else ``cfg.paths.ui``."""
    env = os.environ.get("OPHBENCH_UI_DIR", "").strip()
    root = Path(env).resolve() if env else Path(cfg.paths.ui)
    return SimpleNamespace(root=root, data=root / "data", samples=root / "data" / "samples", media=root / "media")


def media_url(sample_id: str, name: str) -> str:
    return f"media/{sample_id}/{name}"


# ------------------------------------------------------------------------------------ loading
def load_manifest(cfg: Config) -> Optional[list[SampleRecord]]:
    """SampleRecords in manifest order, or None when the manifest does not exist."""
    path = Path(cfg.paths.sample) / "sample_manifest.jsonl"
    if not path.exists():
        return None
    out: list[SampleRecord] = []
    for row in iter_jsonl(path):
        try:
            out.append(SampleRecord.from_dict(row))
        except Exception:
            continue
    return out


def load_validated(cfg: Config) -> dict[str, dict]:
    """sample_id -> validated row (QASet dict + validation) from qa_validated.jsonl."""
    path = Path(cfg.paths.qa) / "qa_validated.jsonl"
    return {row["sample_id"]: row for row in iter_jsonl(path) if row.get("sample_id")}


def load_prepared(prepared_dir: Path) -> Optional[PreparedSample]:
    path = prepared_dir / "sample.json"
    if not path.exists():
        return None
    return PreparedSample.from_dict(read_json(path))


def load_qa(cfg: Config, sample_id: str, validated: dict[str, dict], log) -> tuple[Optional[QASet], Optional[dict]]:
    """(QASet, validation dict|None). Prefers the validated row unless data/qa/<id>.json is newer."""
    qa_path = Path(cfg.paths.qa) / f"{sample_id}.json"
    row = validated.get(sample_id)
    fresh: Optional[dict] = None
    if qa_path.exists():
        try:
            fresh = read_json(qa_path)
        except Exception as e:
            log.warning("%s: cannot read %s (%s)", sample_id, qa_path.name, e)
    if row is not None:
        stale = (
            fresh is not None
            and (fresh.get("provenance") or {}).get("generated_at")
            and (fresh.get("provenance") or {}).get("generated_at") != (row.get("provenance") or {}).get("generated_at")
        )
        if not stale:
            return QASet.from_dict(row), dict(row.get("validation") or {}) or None
        log.warning("%s: qa_validated.jsonl row is older than %s - exporting unvalidated (re-run `validate`)",
                    sample_id, qa_path.name)
    if fresh is None:
        return None, None
    return QASet.from_dict(fresh), None


# ------------------------------------------------------------------------------------ metadata
def humanize(key: str) -> str:
    k = str(key).strip().replace("-", "_")
    if k.startswith("n_"):
        return "Number of " + k[2:].replace("_", " ")
    if k.startswith("has_"):
        return "Has " + k[4:].replace("_", " ")
    if k.startswith("is_"):
        return "Is " + k[3:].replace("_", " ")
    text = k.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else key


def category_label(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, humanize(cat))


def looks_like_path(s: str) -> bool:
    s = s.strip()
    return bool(_PATH_RE.search(s)) or s.count("/") >= 2 or ("!" in s and "/" in s)


def _fmt_number(v: float) -> str:
    if isinstance(v, int):
        return str(v)
    if v.is_integer():
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def format_value(v: Any) -> Optional[str]:
    """Clinician-friendly string for a label value; None means 'do not show'."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        return _fmt_number(v)
    if isinstance(v, str):
        s = " ".join(v.split())
        if not s or looks_like_path(s):
            return None
        return s if len(s) <= MAX_TEXT_CHARS else s[: MAX_TEXT_CHARS - 3] + "..."
    if isinstance(v, (list, tuple, set)):
        items = [x for x in v if x is not None]
        if not items:
            return None
        if all(isinstance(x, dict) for x in items):
            names: list[str] = []
            for d in items:
                for k in ("name", "label", "phase_name", "operation_name", "operation", "phase"):
                    val = d.get(k)
                    if isinstance(val, str) and val.strip() and val.strip() not in names:
                        names.append(val.strip())
                        break
            if not names:
                return None
            items = names
        parts = [p for p in (format_value(x) for x in items) if p]
        if not parts:
            return None
        shown, extra = parts[:MAX_LIST_ITEMS], max(0, len(parts) - MAX_LIST_ITEMS)
        return ", ".join(shown) + (f" (+{extra} more)" if extra else "")
    if isinstance(v, dict):
        parts = []
        for k, x in v.items():
            if isinstance(x, (dict, list, tuple, set)):
                continue
            fv = format_value(x)
            if fv is not None:
                parts.append(f"{LABEL_NAMES.get(str(k), humanize(str(k)))}: {fv}")
        return "; ".join(parts[:MAX_LIST_ITEMS]) or None
    return format_value(str(v))


def label_value(key: str, value: Any) -> Any:
    """Key-aware tweaks before generic formatting: seconds -> readable duration, shares -> %, 0/1 flags -> yes/no."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if math.isnan(float(value)):
        return None
    if key.endswith(("_s", "_seconds")) and not key.startswith("n_"):
        return fmt_duration(float(value)) if value else None
    if key.endswith(("_share", "_fraction")):
        return f"{100.0 * float(value):.0f}%"
    if key in ("adverse_event", "clear", "truncated") and value in (0, 1):
        return bool(value)
    return value


def build_metadata(rec: SampleRecord, probe: Optional[ProbeInfo]) -> list[dict]:
    """Ordered ``[{label, value}]`` rows: core facts first, then readable label fields, then notes."""
    rows: list[dict] = []

    def add(label: str, value: Any) -> None:
        fv = format_value(value)
        if fv is not None:
            rows.append({"label": label, "value": fv})

    meta = DATASET_META.get(rec.dataset, {})
    add("Dataset", meta.get("title", rec.dataset))
    add("Video ID", rec.video_id)
    add("Procedure", rec.procedure)
    add("Procedure category", category_label(rec.procedure_category))
    dur = (probe.duration_s if probe and probe.duration_s else None) or rec.duration_s
    if dur:
        add("Duration", fmt_duration(dur))
    w = (probe.width if probe else None) or rec.width
    h = (probe.height if probe else None) or rec.height
    if w and h:
        add("Resolution", f"{w} x {h}")
    fps = (probe.fps if probe else None) or rec.fps
    if fps:
        add("Frame rate", f"{float(fps):.3g} fps")
    if rec.source_kind == "concat_clips":
        add("Video composition", f"concatenation of {len(rec.source_paths)} phase clips (see timeline note)")
    for key, value in (rec.labels or {}).items():
        if key in DROP_KEYS:
            continue
        add(LABEL_NAMES.get(key, humanize(key)), label_value(key, value))
    if rec.notes:
        add("Notes", rec.notes)
    return rows


# ------------------------------------------------------------------------------------ sample document
def _round(v: Any, nd: int = 3) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, nd)


def build_segments(rec: SampleRecord) -> list[dict]:
    out = []
    for s in rec.segments or []:
        start, end = _round(s.start_s), _round(s.end_s)
        if start is None or end is None:
            continue
        out.append({"label": str(s.label), "start_s": start, "end_s": max(start, end), "kind": s.kind or "phase"})
    return out


def build_frames(prepared: PreparedSample) -> list[dict]:
    out = []
    for f in prepared.frames or []:
        t = _round(f.t_s, 2)
        if t is not None:
            out.append({"t_s": t, "label": f.label_at_t or ""})
    return out


def _evidence(ev: Any) -> Optional[dict]:
    if not isinstance(ev, dict):
        return None
    start, end = _round(ev.get("start_s")), _round(ev.get("end_s"))
    if start is None and end is None:
        return None
    start = start if start is not None else end
    end = end if end is not None else start
    return {"start_s": start, "end_s": end, "observation": str(ev.get("observation") or "")}


def build_questions(qs: QASet, validation: Optional[dict]) -> list[dict]:
    q_issues = (validation or {}).get("question_issues") or {}
    out = []
    for q in qs.questions:
        d = q.to_dict()
        d["qid"] = str(d.get("qid") or f"q{len(out) + 1}")
        d["question"], d["answer"] = str(d.get("question") or ""), str(d.get("answer") or "")
        d["answer_rationale"], d["why_hard"] = str(d.get("answer_rationale") or ""), str(d.get("why_hard") or "")
        d["evidence_timestamps"] = [e for e in (_evidence(x) for x in (d.get("evidence_timestamps") or [])) if e]
        for key in ("agentic_skills", "tool_plan", "options", "metadata_used"):
            d[key] = [str(x) for x in (d.get(key) or [])]
        conf = _round(d.get("confidence"), 3)
        d["confidence"] = conf if conf is not None else 0.0
        d["validation_issues"] = [str(c) for c in q_issues.get(d["qid"], [])]
        out.append(d)
    return out


def local_file_name(sample_id: str) -> str:
    """File name of the sample's video inside an offline batch package (matched by the UI by name)."""
    return f"{sample_id}.mp4"


def build_sample_doc(rec: SampleRecord, prepared: PreparedSample, qs: QASet, validation: Optional[dict],
                     preview_url: Optional[str], batch: Optional[int] = None) -> dict:
    meta = DATASET_META.get(rec.dataset, {})
    probe = prepared.probe
    dur = (probe.duration_s if probe and probe.duration_s else None) or rec.duration_s
    return {
        "ui_version": UI_SAMPLE_VERSION,
        "sample_id": rec.sample_id,
        "batch": batch,
        "dataset": rec.dataset,
        "dataset_title": meta.get("title", rec.dataset),
        "dataset_blurb": meta.get("blurb", ""),
        "citation": rec.citation or meta.get("citation", ""),
        "license": rec.license or meta.get("license", ""),
        "procedure": rec.procedure,
        "procedure_category": rec.procedure_category,
        "video": {
            "preview": preview_url,
            "local_file": local_file_name(rec.sample_id),
            "contact_sheet": media_url(rec.sample_id, "contact_sheet.jpg"),
            "duration_s": _round(dur, 3) if dur else None,
            "width": (probe.width if probe else None) or rec.width,
            "height": (probe.height if probe else None) or rec.height,
            "fps": _round((probe.fps if probe else None) or rec.fps, 3),
        },
        "metadata": build_metadata(rec, probe),
        "segments": build_segments(rec),
        "frames": build_frames(prepared),
        "timeline_note": prepared.timeline_note or "",
        "video_summary": qs.video_summary or "",
        "questions": build_questions(qs, validation),
        "generator_notes": qs.generator_notes or "",
        "provenance": redact(dict(qs.provenance or {})),
        "validation": {
            "ok": bool((validation or {}).get("ok", True)),
            "issues": [str(c) for c in (validation or {}).get("issues", [])],
            "validated": validation is not None,
        },
    }


# ------------------------------------------------------------------------------------ media
def place_media(src: Path, dst: Path, mode: str) -> bool:
    """Symlink/copy ``src`` to ``dst`` (idempotent). Returns False when ``src`` does not exist."""
    if not src.exists():
        return False
    if mode == "none":
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target = src.resolve()
        if dst.is_symlink():
            if Path(os.readlink(dst)) == target:
                return True
            dst.unlink()
        elif dst.exists():
            dst.unlink()
        os.symlink(target, dst)
        return True
    # copy
    if dst.exists() and not dst.is_symlink() and dst.stat().st_size == src.stat().st_size:
        return True
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return True


def export_media(prepared_dir: Path, prepared: PreparedSample, sample_id: str, media_root: Path, mode: str,
                 log) -> tuple[bool, Optional[str]]:
    """Place the contact sheet (and, unless ``mode`` is ``sheets``, the preview) under ui/media/<id>/.

    Returns ``(contact_sheet_present, preview_url_or_None)``.  ``sheets`` copies only the contact sheet (the
    videos are shared offline); ``none`` places nothing but still reports a preview URL when preview.mp4
    exists (media hosted elsewhere via MEDIA_BASE_URL).
    """
    dst_dir = media_root / sample_id
    sheet_src = prepared_dir / (prepared.contact_sheet or "contact_sheet.jpg")
    sheet_ok = place_media(sheet_src, dst_dir / "contact_sheet.jpg", "copy" if mode == "sheets" else mode)
    if not sheet_ok:
        log.warning("%s: contact sheet missing (%s) - thumbnail will not load", sample_id, sheet_src)
    if mode == "sheets":
        return sheet_ok, None
    preview_src = prepared_dir / (prepared.preview or "preview.mp4")
    preview_url = media_url(sample_id, "preview.mp4") if place_media(preview_src, dst_dir / "preview.mp4", mode) else None
    return sheet_ok, preview_url


# ------------------------------------------------------------------------------------ offline packages (13.2)
def package_source(prepared_dir: Path, prepared: PreparedSample, kind: str) -> tuple[Optional[Path], Optional[str]]:
    """Video file to ship for a sample: ``(path, kind_used)``; preview falls back to source when missing."""
    source = prepared_dir / (prepared.source_path or "source.mp4")
    if kind == "preview":
        preview = prepared_dir / (prepared.preview or "preview.mp4")
        if preview.exists():
            return preview, "preview"
    if source.exists():
        return source.resolve(), "source"
    return None, None


def copy_if_needed(src: Path, dst: Path) -> bool:
    """Copy ``src`` (following symlinks) to ``dst`` unless an identical-size copy exists; returns True when copied."""
    size = src.stat().st_size
    if dst.exists() and not dst.is_symlink() and dst.stat().st_size == size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return True


def package_readme(batch: int, rows: list[dict], total_bytes: int, kind: str) -> str:
    kinds = sorted({r["kind"] for r in rows})
    what = ("360p preview re-encodes of the source videos" if kinds == ["preview"] else
            "original-resolution source videos (or concatenated case videos)" if kinds == ["source"] else
            "360p previews, with original-resolution source videos where no preview existed")
    return "\n".join([
        f"ophbench clinician review - offline video package for batch {batch}",
        f"generated: {now_iso()}   videos: {len(rows)}   size: {human_size(total_bytes)}   content: {what}",
        "",
        "HOW TO USE",
        "1. Open the review website link you received and enter your annotator ID.",
        f"2. On the list page click 'Load videos folder' and select THIS folder ({batch_dir_name(batch)}).",
        "   The browser reads the files directly from your disk; nothing is uploaded anywhere.",
        "3. Open a sample: its video now plays from this folder (timeline clicks and evidence chips seek it).",
        "   If a video is missing the page shows the contact sheet (sampled frames with timestamps) instead.",
        "",
        "DO NOT RENAME the files: the website matches them by name (<sample_id>.mp4).",
        "MANIFEST.csv lists every file with its dataset, procedure and duration.",
        "The videos are research data distributed under the licences of the source datasets (see the",
        "'About the dataset' panel in the website); do not share them further.",
        "",
    ])


def write_package(pkg_dir: Path, items: list[tuple[SampleRecord, PreparedSample, Path]], kind: str, batch: int,
                  log) -> dict:
    """Write ``<pkg_dir>/<sample_id>.mp4`` for every item plus MANIFEST.csv and README.txt."""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    total = 0
    n_copied = n_fallback = n_missing = 0
    for rec, prepared, prepared_dir in items:
        src, used = package_source(prepared_dir, prepared, kind)
        if src is None:
            log.warning("%s: no %s video to package (prepared folder has neither preview.mp4 nor source.mp4)",
                        rec.sample_id, kind)
            n_missing += 1
            continue
        if used != kind:
            n_fallback += 1
            log.warning("%s: no preview.mp4 - packaging the source video instead (run `prepare` to add previews)",
                        rec.sample_id)
        dst = pkg_dir / local_file_name(rec.sample_id)
        try:
            n_copied += int(copy_if_needed(src, dst))
        except OSError as e:
            log.warning("%s: could not copy %s -> %s (%s)", rec.sample_id, src, dst, e)
            n_missing += 1
            continue
        size = dst.stat().st_size
        total += size
        dur = prepared.probe.duration_s if prepared.probe and prepared.probe.duration_s else rec.duration_s
        rows.append({"sample_id": rec.sample_id, "dataset": rec.dataset, "procedure": rec.procedure,
                     "duration_s": round(float(dur), 3) if dur else "", "file": dst.name,
                     "size_mb": round(size / 1e6, 1), "batch": batch, "kind": used})
    write_csv(pkg_dir / "MANIFEST.csv", rows, PACKAGE_MANIFEST_FIELDS)
    (pkg_dir / "README.txt").write_text(package_readme(batch, rows, total, kind), encoding="utf-8")
    log.info("offline package %s: %d video(s), %s (%d newly copied, %d source fallbacks, %d missing)", pkg_dir,
             len(rows), human_size(total), n_copied, n_fallback, n_missing)
    return {"dir": str(pkg_dir), "n": len(rows), "bytes": total, "kind": kind, "n_copied": n_copied,
            "n_fallback_source": n_fallback, "n_missing": n_missing}


# ------------------------------------------------------------------------------------ assignments
def parse_list_arg(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    out: list[str] = []
    for v in value:
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
    return out


def build_assignments(sample_ids: list[str], annotators: list[str], overlap_fraction: float, seed: int) -> dict:
    """Round-robin over annotators; ``ceil(overlap*n)`` seeded-random samples go to everyone."""
    annotators = parse_list_arg(annotators)
    ids = list(sample_ids)
    if not annotators:
        return {"annotators": [], "by_annotator": {}, "overlap": []}
    if len(annotators) == 1:
        return {"annotators": annotators, "by_annotator": {annotators[0]: ids}, "overlap": []}
    frac = max(0.0, min(1.0, float(overlap_fraction or 0.0)))
    k = min(len(ids), math.ceil(frac * len(ids))) if frac > 0 else 0
    shuffled = list(ids)
    random.Random(int(seed)).shuffle(shuffled)
    overlap = set(shuffled[:k])
    by: dict[str, list[str]] = {a: [] for a in annotators}
    i = 0
    for sid in ids:
        if sid in overlap:
            for a in annotators:
                by[a].append(sid)
        else:
            by[annotators[i % len(annotators)]].append(sid)
            i += 1
    return {"annotators": annotators, "by_annotator": by, "overlap": [s for s in ids if s in overlap]}


def merge_assignments(previous: Optional[dict], new: dict, replaced_ids: set[str], log=None) -> dict:
    """Additive batch export: keep earlier batches' assignments, replace those of ``replaced_ids`` with ``new``."""
    prev = previous if isinstance(previous, dict) else {}
    annotators = list(new.get("annotators") or [])
    prev_by = prev.get("by_annotator") if isinstance(prev.get("by_annotator"), dict) else {}
    dropped = sorted(a for a in prev_by if a not in annotators)
    if dropped and log is not None:
        log.warning("annotator(s) %s of earlier batches are not in the current list and were dropped", dropped)
    by: dict[str, list[str]] = {}
    for a in annotators:
        kept = [sid for sid in (prev_by.get(a) or []) if sid not in replaced_ids]
        by[a] = kept + [sid for sid in (new.get("by_annotator") or {}).get(a, []) if sid not in kept]
    overlap = [sid for sid in (prev.get("overlap") or []) if sid not in replaced_ids] + list(new.get("overlap") or [])
    return {"annotators": annotators, "by_annotator": by, "overlap": overlap}


# ------------------------------------------------------------------------------------ stage entry
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--media-mode", choices=MEDIA_MODES, default=None,
                   help="how to place media under ui/media: symlink (default without --batch) | copy | none | "
                        "sheets (contact sheets only; default with --batch, videos are shared offline)")
    p.add_argument("--annotators", default=None,
                   help="comma-separated annotator ids (default: config ui.annotators; empty = everyone sees all)")
    p.add_argument("--overlap", type=float, default=None,
                   help="fraction of samples assigned to every annotator for agreement (default: ui.overlap_fraction)")
    p.add_argument("--include-invalid", action="store_true",
                   help="accepted for compatibility: items failing validation are always exported and flagged "
                        "(has_issues / validation_issues)")
    p.add_argument("--batch", type=int, default=None,
                   help="export only annotation batch K (1-based, canonical benchmark order); additive over earlier "
                        "batches and writes the offline package ui_packages/batch_<K>/")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="samples per batch (default: config `batches.size`, 100)")
    p.add_argument("--no-package", action="store_true", dest="no_package",
                   help="with --batch: do not write the offline video package")
    p.add_argument("--package-kind", choices=PACKAGE_KINDS, default=None, dest="package_kind",
                   help="offline package content: preview (360p, default from config) | source (original video)")
    p.add_argument("--packages-dir", default=None, dest="packages_dir",
                   help="where offline packages are written (default: $OPHBENCH_PACKAGES_DIR or <repo>/ui_packages)")


def _read_json_or(path: Path, default: Any, log) -> Any:
    if not path.exists():
        return default
    try:
        return read_json(path)
    except Exception as e:  # noqa: BLE001 - a corrupt previous export must not block the new one
        log.warning("could not read %s (%s); ignoring it", path, e)
        return default


def main(args: argparse.Namespace, cfg: Config) -> int:
    cfg.ensure_dirs()
    log = get_logger(STAGE, cfg)
    started = now_iso()
    paths = ui_paths(cfg)
    batch = getattr(args, "batch", None)
    size = batch_size(cfg, getattr(args, "batch_size", None))
    mode = getattr(args, "media_mode", None) or ("sheets" if batch is not None else "symlink")
    if mode not in MEDIA_MODES:
        log.error("unknown --media-mode %r (choose from %s)", mode, "|".join(MEDIA_MODES))
        return 1
    if batch is not None and int(batch) < 1:
        log.error("--batch must be >= 1 (got %s)", batch)
        return 1
    try:
        kind = package_kind(cfg, getattr(args, "package_kind", None))
    except ValueError as e:
        log.error("%s", e)
        return 1
    annotators = parse_list_arg(getattr(args, "annotators", None)) or parse_list_arg(cfg.get("ui.annotators", []))
    overlap_arg = getattr(args, "overlap", None)
    overlap = float(overlap_arg) if overlap_arg is not None else float(cfg.get("ui.overlap_fraction", 0.1) or 0.0)

    records = load_manifest(cfg)
    if records is None:
        log.error("sample manifest not found at %s - run `sample` first", Path(cfg.paths.sample) / "sample_manifest.jsonl")
        return 1
    ordered = order_samples(records)
    rank_by_id = ranks(records)
    batch_by_id = assign_batches(records, size)
    total_batches = n_batches(len(records), size)
    if batch is not None:
        selected = select_batch(ordered, int(batch), size)
        log.info("batch %d/%d (size %d): %d manifest sample(s)", batch, total_batches, size, len(selected))
        if not selected:
            log.warning("batch %d is empty (manifest has %d samples -> %d batches of %d)", batch, len(records),
                        total_batches, size)
    else:
        selected = ordered
    validated = load_validated(cfg)
    if not validated:
        log.warning("no validated QA (data/qa/qa_validated.jsonl empty or missing) - falling back to data/qa/<id>.json; "
                    "run `validate` to get issue flags")
    paths.samples.mkdir(parents=True, exist_ok=True)
    if mode != "none":
        paths.media.mkdir(parents=True, exist_ok=True)
    log.info("exporting %d manifest sample(s) -> %s (media mode: %s, annotators: %s, overlap: %.2f%s)",
             len(selected), paths.root, mode, annotators or "everyone", overlap,
             f", batch {batch}" if batch is not None else "")

    counts: Counter = Counter()
    entries: list[dict] = []
    packaged: list[tuple[SampleRecord, PreparedSample, Path]] = []
    for rec in selected:
        sid = rec.sample_id
        prepared_dir = resolve_prepared_dir(cfg, sid)
        try:
            prepared = load_prepared(prepared_dir)
        except Exception as e:
            log.warning("%s: unreadable sample.json (%s: %s) - skipped", sid, type(e).__name__, e)
            counts["skipped_bad_prepared"] += 1
            continue
        if prepared is None:
            log.debug("%s: not prepared (no %s) - skipped", sid, prepared_dir / "sample.json")
            counts["skipped_not_prepared"] += 1
            continue
        try:
            qs, validation = load_qa(cfg, sid, validated, log)
        except Exception as e:
            log.warning("%s: unreadable QA (%s: %s) - skipped", sid, type(e).__name__, e)
            counts["skipped_bad_qa"] += 1
            continue
        if qs is None:
            log.debug("%s: no QA yet - skipped", sid)
            counts["skipped_no_qa"] += 1
            continue
        if not qs.questions:
            log.warning("%s: QA set has no questions - skipped", sid)
            counts["skipped_no_questions"] += 1
            continue
        sheet_ok, preview_url = export_media(prepared_dir, prepared, sid, paths.media, mode, log)
        doc = build_sample_doc(rec, prepared, qs, validation, preview_url, batch=batch_by_id.get(sid))
        write_json(paths.samples / f"{sid}.json", doc)
        has_issues = validation is not None and not bool(validation.get("ok", True))
        entries.append({
            "sample_id": sid,
            "batch": batch_by_id.get(sid),
            "dataset": rec.dataset,
            "dataset_title": doc["dataset_title"],
            "procedure": rec.procedure,
            "procedure_category": rec.procedure_category,
            "duration_s": doc["video"]["duration_s"],
            "n_questions": len(doc["questions"]),
            "thumb": media_url(sid, "contact_sheet.jpg"),
            "preview": preview_url,
            "assigned_to": [],
            "has_issues": has_issues,
        })
        packaged.append((rec, prepared, prepared_dir))
        counts["exported"] += 1
        counts["has_issues"] += int(has_issues)
        counts["with_preview"] += int(preview_url is not None)
        counts["missing_contact_sheet"] += int(not sheet_ok)
        counts["unvalidated"] += int(validation is None)
    if counts["skipped_not_prepared"] or counts["skipped_no_qa"]:
        log.warning("skipped %d sample(s) without a prepared folder and %d without a QA set (details at DEBUG level; "
                    "run `prepare` / `generate` for them)", counts["skipped_not_prepared"], counts["skipped_no_qa"])

    # ---- merge with earlier batches (additive) ------------------------------------------------
    new_ids = [e["sample_id"] for e in entries]
    if batch is not None:
        this_batch_ids = {sid for sid, b in batch_by_id.items() if b == batch}
        prev_index = _read_json_or(paths.data / "index.json", [], log)
        kept = [e for e in prev_index if isinstance(e, dict) and e.get("sample_id") in rank_by_id
                and e.get("sample_id") not in this_batch_ids]
        for e in kept:
            e["batch"] = batch_by_id.get(e["sample_id"])
        n_stale = len(prev_index) - len(kept) - sum(1 for e in prev_index if isinstance(e, dict) and e.get("sample_id") in this_batch_ids)
        if n_stale:
            log.info("dropped %d earlier index entries that are no longer in the manifest", n_stale)
        all_entries = sorted(kept + entries, key=lambda e: rank_by_id.get(e["sample_id"], 1 << 30))
        prev_assign = _read_json_or(paths.data / "assignments.json", None, log)
        new_assign = build_assignments(new_ids, annotators, overlap, cfg.seed + int(batch))
        assignments = merge_assignments(prev_assign, new_assign, this_batch_ids, log) if prev_assign else new_assign
        prev_meta = _read_json_or(paths.data / "meta.json", {}, log)
        batches_meta = dict((prev_meta or {}).get("batches") or {})
    else:
        this_batch_ids = set(rank_by_id)
        all_entries = entries
        assignments = build_assignments(new_ids, annotators, overlap, cfg.seed)
        batches_meta = {}
    membership = {a: set(s) for a, s in assignments["by_annotator"].items()}
    for e in all_entries:
        e["assigned_to"] = [a for a in assignments["annotators"] if e["sample_id"] in membership.get(a, set())]
    all_ids = [e["sample_id"] for e in all_entries]

    # ---- offline package -------------------------------------------------------------------------
    package_info: Optional[dict] = None
    if batch is not None and packaged and not getattr(args, "no_package", False):
        pkg_dir = packages_dir(cfg, getattr(args, "packages_dir", None)) / batch_dir_name(int(batch))
        try:
            package_info = write_package(pkg_dir, packaged, kind, int(batch), log)
        except Exception as e:  # noqa: BLE001 - the UI export itself succeeded; report and continue
            log.error("offline package failed: %s: %s", type(e).__name__, e)
            package_info = {"dir": str(pkg_dir), "error": f"{type(e).__name__}: {e}"}
    if batch is not None:
        batches_meta[str(int(batch))] = {"n": len(entries), "exported_at": now_iso(), "size": size,
                                         "package": (package_info or {}).get("dir"), "package_kind": kind if package_info else None,
                                         "n_videos_packaged": (package_info or {}).get("n")}

    datasets_present = dict(sorted(Counter(e["dataset"] for e in all_entries).items()))
    dataset_meta = {}
    for ds in datasets_present:
        m = DATASET_META.get(ds)
        rec0 = next((r for r in records if r.dataset == ds), None)
        dataset_meta[ds] = {
            "title": (m or {}).get("title", ds), "blurb": (m or {}).get("blurb", ""),
            "license": (m or {}).get("license", rec0.license if rec0 else ""),
            "citation": (m or {}).get("citation", rec0.citation if rec0 else ""),
        }
    meta = {
        "generated_at": now_iso(),
        "ui_version": UI_SAMPLE_VERSION,
        "n_samples": len(all_entries),
        "datasets": datasets_present,
        "select_min": int(cfg.get("ui.questions_to_select_min", 1) or 1),
        "select_max": int(cfg.get("ui.questions_to_select_max", 2) or 2),
        "annotators": assignments["annotators"],
        "dataset_meta": dataset_meta,
        "batches": dict(sorted(batches_meta.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0)),
        "batch_size": size,
        "n_batches_total": total_batches,
        "offline_media": batch is not None,
        "n_with_issues": sum(1 for e in all_entries if e.get("has_issues")),
        "n_unvalidated": counts["unvalidated"],
        "media_mode": mode,
        "overlap_fraction": overlap,
        "ui_config": redact(dict(cfg.section("ui"))),
        "pipeline": {"seed": cfg.seed, "model": cfg.get("llm.model"), "reasoning_effort": cfg.get("llm.reasoning_effort"),
                     "prompt_version": cfg.get("llm.prompt_version")},
    }
    write_json(paths.data / "index.json", all_entries)
    write_json(paths.data / "assignments.json", assignments)
    write_json(paths.data / "meta.json", meta)

    keep = set(all_ids)
    if batch is None:
        stale = [p for p in paths.samples.glob("*.json") if p.stem not in keep]
    else:  # additive: only this batch's samples that were not exported now are stale
        stale = [p for p in paths.samples.glob("*.json") if p.stem in this_batch_ids and p.stem not in keep]
    for p in stale:
        p.unlink()
    if stale:
        log.info("removed %d stale sample file(s) from %s", len(stale), paths.samples)

    if not entries:
        log.warning("nothing exported: no %ssample has both a prepared folder and a QA set yet",
                    f"batch-{batch} " if batch is not None else "manifest ")
    summary = {**dict(counts), "n_manifest": len(records), "n_selected": len(selected), "n_exported": len(entries),
               "n_index_total": len(all_entries), "batch": batch, "batch_size": size, "annotators": assignments["annotators"],
               "n_overlap": len(assignments["overlap"]), "media_mode": mode, "ui_dir": str(paths.root),
               "datasets": datasets_present, "package": package_info}
    log.info("exported %d/%d sample(s) (%d with issues, %d with preview, %d unvalidated); index now holds %d; skipped: "
             "%d not prepared, %d without QA, %d without questions; annotators=%s overlap=%d",
             len(entries), len(selected), counts["has_issues"], counts["with_preview"], counts["unvalidated"],
             len(all_entries), counts["skipped_not_prepared"] + counts["skipped_bad_prepared"],
             counts["skipped_no_qa"] + counts["skipped_bad_qa"], counts["skipped_no_questions"],
             assignments["annotators"] or "everyone", len(assignments["overlap"]))
    log.info("UI data: %s  media: %s  (serve ui/ locally with scripts/serve_ui.py or push ui/ to GitHub Pages)",
             paths.data, paths.media)
    record_run(cfg, STAGE, args, summary, started_at=started)
    return 0
