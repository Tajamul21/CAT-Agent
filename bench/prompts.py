"""Prompt construction for the generation stage (DESIGN.md §7).

``build_messages(prepared, cfg)`` returns ``(system_text, user_parts)`` where ``user_parts`` is an
ordered list of ``{"type": "text", "text": ...}`` and ``{"type": "image", "path": ..., "caption": ...}``
items; ``bench.llm_client.GatewayClient`` converts them to the route specific shapes.  The texts come
from ``config/prompts/system.md`` and ``config/prompts/user_template.md`` (``{field}`` placeholders,
rendered with a tolerant substitution so literal braces survive).  ``render_text_only`` produces the
text-only rendering written by ``generate --dry-run``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from bench.config import Config
from bench.schema import DATASET_META, PROMPT_VERSION, FrameInfo, PreparedSample, Segment, VideoRecord
from bench.util import fmt_duration, fmt_time, md_table

FRAMES_MARKER = "<<FRAMES>>"
_FIELD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_LOG = logging.getLogger("ophbench.prompts")

#: label keys rendered with a clinician-friendly name (order = display order)
LABEL_DISPLAY: list[tuple[str, str]] = [
    ("site", "Recording site code"),
    ("site_name", "Recording site"),
    ("microscope", "Microscope"),
    ("surgeon_id", "Surgeon id (anonymised)"),
    ("surgeon", "Surgeon id (anonymised)"),
    ("experience", "Surgeon experience level"),
    ("operation_type", "Operation type code"),
    ("operation_type_expanded", "Operation type"),
    ("operation_type_glossary", "Operation type codes"),
    ("operation_components", "Operation components"),
    ("goniotomy_degrees", "Goniotomy extent (degrees)"),
    ("combined_with_phaco", "Combined with phacoemulsification"),
    ("knife", "Knife code"),
    ("knife_description", "Knife"),
    ("gt_incision_num", "Number of goniotomy incisions (ground truth)"),
    ("gt_incision_note_en", "Incision note"),
    ("primary_surgery", "Primary surgery label"),
    ("surgery_names", "All surgery labels"),
    ("n_phases", "Distinct annotated phases"),
    ("n_clips", "Phase clips concatenated"),
    ("truncated", "Case truncated to fit the duration cap"),
    ("full_case_duration_s", "Full case duration before truncation (s)"),
    ("annotated", "Phase annotations available"),
    ("flags", "Usage flags (from annotations)"),
    ("has_suture", "Suture phase present"),
    ("has_not_cataract", "Non-standard / not-cataract segment present"),
    ("idle_share", "Idle share of the video"),
    ("skill_scores", "Expert skill scores (1-5, higher is better)"),
    ("microscope_use", "Skill - microscope use (1-5)"),
    ("instrument_handling", "Skill - instrument handling (1-5)"),
    ("tissue_handling", "Skill - tissue handling (1-5)"),
    ("motion", "Skill - motion (1-5)"),
    ("commencement_of_flap", "Skill - commencement of flap (1-5)"),
    ("circular_completion", "Skill - circular completion (1-5)"),
    ("averaged", "Skill - averaged score (1-5)"),
    ("skill_tertile", "Skill tertile within dataset"),
    ("phase", "Phase shown in this clip"),
    ("repeated_phases", "Phases that occur more than once"),
    ("n_distinct_phases", "Distinct phases"),
    ("adverse_event", "Adverse event flagged by annotators"),
    ("adverse_event_comment", "Adverse event comment"),
    ("comment", "Annotator comment"),
    ("instruction", "Narration-derived caption (dataset 'instruction')"),
    ("category_keywords", "Matched procedure keywords"),
    ("split", "Dataset split"),
]
#: label keys that are bulky, redundant with other blocks, or meaningless to the model
SKIP_LABELS: set[str] = {
    # media / bookkeeping
    "frames", "fps", "width", "height", "duration_s", "file_size_mb", "frame_count", "nominal_resolution",
    "nominal_fps", "nominal_duration_s", "annotation_fps", "video_number", "json_video_id", "case_id", "raw_index",
    "subset_index", "raw_video_id", "phase_video_id", "metadata_filename", "file_name", "loose_file", "subset",
    "split", "in_phase_subset", "in_filtered_28k", "source_video_id", "clip_index", "instruction_words",
    "size_bin", "duration_bin", "experience_code", "primary_surgery_id", "surgery_ids", "phase_code",
    # redundant with the timeline table / glossary blocks
    "original_segments", "clips", "operations", "phase_sequence", "phase_names", "step_names", "n_segments",
    "n_phase_segments", "n_flag_segments", "has_overlapping_segments", "idle_seconds", "unannotated_lead_s",
    "phase_glossary", "operation_glossary", "clip_window", "clip_overlap_max_s", "full_case_n_clips",
    "full_case_n_phases", "knife_norm", "manifest_adverse_event",
    # file system paths
    "tracking_annotation_zip", "annotation_csv", "clips_dir",
}
#: the six individual skill indicators (hidden when the ``skill_scores`` dict is present)
SKILL_KEYS: tuple[str, ...] = ("microscope_use", "instrument_handling", "tissue_handling", "motion",
                               "commencement_of_flap", "circular_completion")

#: dataset specific notes appended to the dataset block (interpretation hints for the model)
DATASET_PROMPT_NOTES: dict[str, str] = {
    "cataract101": "Phase labels are expert annotations of 10 quasi-standard phacoemulsification phases; "
                   "unannotated stretches are idle/transition time. Surgeon experience: 'low' = senior resident "
                   "/ early career, 'high' = experienced surgeon.",
    "cataract1k": "Phase labels (when present) were annotated at 1 fps; 'Idle' covers pauses and transitions. "
                  "Usage flags (trypan blue, iris hooks, Malyugin ring) mark adjuncts used during the case. "
                  "Videos without phase annotations carry no timeline.",
    "lmm_phase": "Complete procedures with frame-accurate boundaries for 13 phases including Idle. "
                 "Site S1 = Farabi (720x480 @ 30 fps), S2 = Noor (1080p @ 60 fps).",
    "lmm_skill": "A single capsulorhexis phase clip. Expert skill indicators are 1-5 (higher is better): microscope "
                 "use, instrument handling, tissue handling, motion, commencement of flap, circular completion; "
                 "'adverse event' = 1 when the annotators flagged a complication during the rhexis.",
    "lmm_raw": "Unannotated complete cataract procedure: there is no label timeline; derive everything from the frames.",
    "migs": "Step labels may overlap (Gonioscopy contains Goniotomy). Operation type codes: GT120/GT240/GT360 = "
            "goniotomy over 120/240/360 degrees; PEI = phacoemulsification with IOL implantation; GSL = "
            "goniosynechialysis; GATT = gonioscopy-assisted transluminal trabeculotomy; SPI = (likely) surgical "
            "peripheral iridectomy. 'Would closure' in the source labels means wound closure.",
    "ophnet": "The displayed video is the concatenation of the trimmed phase clips of one case, so cuts between "
              "phases are abrupt and idle time between phases is removed. Phase names follow the OphNet-2024 "
              "taxonomy (glossary below when available).",
    "ophora": "A short (~5 s) clip from a narrated YouTube surgical video; the 'instruction' caption was derived from "
              "the narration and may be imprecise. There is no phase timeline; questions must rely on what is visible.",
}


# ------------------------------------------------------------------------------------------ files
@lru_cache(maxsize=8)
def _read_prompt_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompt file missing: {p} (expected config/prompts/system.md and user_template.md)")
    return p.read_text(encoding="utf-8")


def load_prompt_texts(cfg: Config) -> tuple[str, str]:
    """``(system.md, user_template.md)`` raw texts."""
    d = Path(cfg.paths.prompts)
    return _read_prompt_file(str(d / "system.md")), _read_prompt_file(str(d / "user_template.md"))


def prompt_version(cfg: Config) -> str:
    return str(cfg.get("llm.prompt_version", PROMPT_VERSION) or PROMPT_VERSION)


def prompt_fingerprint(cfg: Config) -> str:
    """Short sha1 of both prompt files - stored in provenance so prompt edits are traceable."""
    sys_t, usr_t = load_prompt_texts(cfg)
    return hashlib.sha1((sys_t + "\n---\n" + usr_t).encode("utf-8")).hexdigest()[:12]


def render_template(template: str, mapping: dict[str, Any]) -> str:
    """Substitute ``{name}`` fields present in ``mapping``; leave any other braces untouched."""
    def sub(m: re.Match) -> str:
        key = m.group(1)
        return str(mapping[key]) if key in mapping else m.group(0)
    return _FIELD_RE.sub(sub, template)


# ------------------------------------------------------------------------------------------ pieces
def frame_caption(k: int, n: int, t_s: float, label: Optional[str]) -> str:
    """``Frame k/n - t=MM:SS.s - <label>``."""
    return f"Frame {k}/{n} - t={fmt_time(t_s)} - {label or 'no phase label'}"


def label_at(record: VideoRecord, t_s: float) -> Optional[str]:
    """Phase/step label covering ``t_s`` (first match in start order; flags excluded)."""
    best: Optional[Segment] = None
    for s in sorted(record.phase_segments, key=lambda x: (x.start_s, x.end_s)):
        if s.start_s <= t_s < s.end_s or (s.start_s <= t_s <= s.end_s and s.end_s == s.start_s):
            if best is None or s.duration_s < best.duration_s:  # prefer the most specific (shortest) segment
                best = s
    return best.label if best else None


def _fmt_value(v: Any) -> Optional[str]:
    """Render a label value compactly; None when it is too bulky to show."""
    if v is None or v == "" or v == [] or v == {}:
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1 else f"{v:.4g}"
    if isinstance(v, (int, str)):
        s = str(v).strip()
        return s[:600] if s else None
    if isinstance(v, (list, tuple)):
        if len(v) <= 40 and all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
            return ", ".join(str(x) for x in v)
        return f"{len(v)} items (not shown)"
    if isinstance(v, dict):
        if len(v) <= 20 and all(isinstance(x, (str, int, float, bool)) or x is None for x in v.values()):
            return "; ".join(f"{str(k).replace('_', ' ')}: {_fmt_value(x) if x is not None else 'n/a'}" for k, x in v.items())
        return f"{len(v)} fields (not shown)"
    return str(v)[:300]


def _pretty_key(k: str) -> str:
    return k.replace("_", " ").strip().capitalize()


def metadata_block(prepared: PreparedSample) -> str:
    """Bullet list: procedure, category, duration, resolution and the record's informative labels."""
    rec, probe = prepared.record, prepared.probe
    duration = probe.duration_s if probe and probe.duration_s else rec.duration_s
    w, h = (probe.width or rec.width), (probe.height or rec.height)
    fps = probe.fps or rec.fps
    lines = [
        f"- Sample id: {rec.sample_id}",
        f"- Procedure: {rec.procedure or 'unknown'}",
        f"- Procedure category: {rec.procedure_category}",
        f"- Displayed video duration: {fmt_duration(duration)}" + (f" ({fmt_time(duration)})" if duration else ""),
    ]
    if w and h:
        lines.append(f"- Resolution: {w}x{h}" + (f" @ {float(fps):.4g} fps" if fps else ""))
    if rec.source_kind == "concat_clips":
        n = len(prepared.concat_map or []) or rec.labels.get("n_clips")
        lines.append(f"- Source: {n or 'several'} phase clips concatenated into one video (see timeline note)")
    labels = rec.labels or {}
    shown: set[str] = set()
    if isinstance(labels.get("skill_scores"), dict):
        shown.update(SKILL_KEYS)
    for key, name in LABEL_DISPLAY:
        if key in labels and key not in shown:
            raw_val = labels[key]
            if key == "adverse_event" and raw_val in (0, 1, "0", "1"):
                raw_val = bool(int(raw_val))
            val = _fmt_value(raw_val)
            if val is not None and not _looks_like_path(val):
                lines.append(f"- {name}: {val}")
            shown.add(key)
    for key in sorted(labels):
        if key in shown or key in SKIP_LABELS or key.startswith("_"):
            continue
        val = _fmt_value(labels[key])
        if val is not None and not _looks_like_path(val):
            lines.append(f"- {_pretty_key(key)}: {val}")
    ops = labels.get("operations")
    if isinstance(ops, list) and ops and not _operations_have_times(ops):
        names = _operation_names(ops)
        if names:
            lines.append(f"- Finer operations annotated (in order, times not shown): {', '.join(names)}")
    if rec.notes:
        lines.append(f"- Processing notes: {'; '.join(str(n) for n in rec.notes[:5])}")
    return "\n".join(lines)


def _looks_like_path(val: str) -> bool:
    return isinstance(val, str) and (val.startswith("/") or val.startswith("\\\\")) and " " not in val.strip()


_OP_START_KEYS = ("concat_start_s", "start_s", "start")
_OP_END_KEYS = ("concat_end_s", "end_s", "end")
_OP_NAME_KEYS = ("operation_name", "name", "operation", "label")


def _operations_have_times(ops: list) -> bool:
    return bool(ops) and all(isinstance(o, dict) and any(k in o for k in _OP_START_KEYS) and any(k in o for k in _OP_END_KEYS)
                             for o in ops)


def _first(o: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in o and o[k] is not None:
            return o[k]
    return None


def operations_table(labels: dict, limit: int = 80) -> str:
    """OphNet finer operations as ``name | start | end | duration`` in the displayed (concatenated) timeline."""
    ops = labels.get("operations") if isinstance(labels, dict) else None
    if not isinstance(ops, list) or not _operations_have_times(ops):
        return ""
    rows = []
    for o in sorted(ops, key=lambda x: float(_first(x, _OP_START_KEYS) or 0.0)):
        name = _first(o, _OP_NAME_KEYS)
        st, en = _first(o, _OP_START_KEYS), _first(o, _OP_END_KEYS)
        if name is None or st is None or en is None:
            continue
        rows.append([str(name), fmt_time(float(st)), fmt_time(float(en)), fmt_time(max(0.0, float(en) - float(st)))])
    if not rows:
        return ""
    extra = f" (first {limit} of {len(rows)})" if len(rows) > limit else ""
    return f"**Finer operations** ({len(rows)}){extra}\n" + md_table(["name", "start", "end", "duration"], rows[:limit])


def _operation_names(ops: Any, limit: int = 60) -> list[str]:
    if not isinstance(ops, list):
        return []
    names: list[str] = []
    for o in ops:
        if isinstance(o, dict):
            for k in ("operation_name", "name", "operation", "label"):
                if isinstance(o.get(k), str):
                    names.append(o[k])
                    break
        elif isinstance(o, str):
            names.append(o)
    out: list[str] = []
    for n in names:
        if not out or out[-1] != n:
            out.append(n)
    return out[:limit]


_KIND_TITLES = {"phase": "Phases", "step": "Steps", "operation": "Operations", "flag": "Usage flags"}


def timeline_tables(prepared: PreparedSample) -> str:
    """Markdown table(s) ``name | start | end | duration`` for the record's segments (grouped by kind)."""
    segs = prepared.record.segments or []
    if not segs:
        return ("No expert label timeline is available for this video. Build the timeline yourself from the frames "
                "and state your uncertainty.")
    by_kind: dict[str, list[Segment]] = {}
    for s in segs:
        by_kind.setdefault(s.kind or "phase", []).append(s)
    blocks: list[str] = []
    for kind in sorted(by_kind, key=lambda k: list(_KIND_TITLES).index(k) if k in _KIND_TITLES else 99):
        rows = [[s.label, fmt_time(s.start_s), fmt_time(s.end_s), fmt_time(s.duration_s)]
                for s in sorted(by_kind[kind], key=lambda x: (x.start_s, x.end_s))]
        title = _KIND_TITLES.get(kind, kind.capitalize())
        header = f"**{title}** ({len(rows)})" if len(by_kind) > 1 else f"{len(rows)} labelled segments"
        blocks.append(header + "\n" + md_table(["name", "start", "end", "duration"], rows))
    ops_table = operations_table(prepared.record.labels or {})
    if ops_table:
        if len(blocks) == 1:
            blocks[0] = blocks[0].replace(f"{len(segs)} labelled segments", f"**Phases** ({len(segs)})", 1)
        blocks.append(ops_table)
    return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def _ophnet_labels(resources_dir: str) -> dict:
    p = Path(resources_dir) / "ophnet_labels.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def phase_glossary(prepared: PreparedSample, resources_dir: Optional[Path] = None) -> str:
    """``name: description`` lines for labels that carry descriptions (segment extras / OphNet taxonomy)."""
    rec = prepared.record
    entries: dict[str, str] = {}
    oph = _ophnet_labels(str(resources_dir)) if (resources_dir and rec.dataset == "ophnet") else {}
    for s in rec.segments or []:
        if s.label in entries:
            continue
        desc = None
        if isinstance(s.extra, dict):
            desc = s.extra.get("description") or s.extra.get("meaning")
        if not desc and oph and s.label_id is not None:
            table = oph.get("operation_description" if s.kind == "operation" else "phase_description", {})
            desc = table.get(str(s.label_id))
        if desc:
            entries[s.label] = str(desc).strip()
    labels = rec.labels if isinstance(rec.labels, dict) else {}
    for key in ("phase_glossary", "phase_descriptions", "operation_glossary"):
        extra = labels.get(key)
        if isinstance(extra, dict):
            for k, v in extra.items():
                if isinstance(v, str) and v.strip() and str(k) not in entries:
                    entries[str(k)] = v.strip()
    if not entries:
        return ""
    lines = [f"- {k}: {v}" for k, v in list(entries.items())[:60]]
    return "### Label glossary\n" + "\n".join(lines)


# ------------------------------------------------------------------------------------------ build
def _resolve_frames(prepared: PreparedSample, sample_dir: Path, log: Optional[logging.Logger]) -> list[tuple[FrameInfo, Path]]:
    out: list[tuple[FrameInfo, Path]] = []
    missing = 0
    for f in sorted(prepared.frames, key=lambda x: (x.t_s, x.idx)):
        p = Path(f.file)
        if not p.is_absolute():
            p = sample_dir / p
        if p.exists():
            out.append((f, p))
        else:
            missing += 1
    if missing:
        (log or _LOG).warning("%s: %d frame file(s) missing under %s - skipped", prepared.record.sample_id, missing, sample_dir)
    return out


def _mapping(prepared: PreparedSample, cfg: Config, n_frames: int) -> dict[str, Any]:
    rec = prepared.record
    meta = DATASET_META.get(rec.dataset, {})
    glossary = phase_glossary(prepared, Path(cfg.paths.resources))
    note = (prepared.timeline_note or "").strip()
    n_q = int(cfg.get("llm.questions_per_video", 3) or 3)
    return {
        "dataset_title": meta.get("title", rec.dataset),
        "dataset_blurb": meta.get("blurb", ""),
        "license": rec.license or meta.get("license", "unknown"),
        "citation": rec.citation or meta.get("citation", ""),
        "dataset_notes": f"Notes: {DATASET_PROMPT_NOTES[rec.dataset]}" if rec.dataset in DATASET_PROMPT_NOTES else "",
        "metadata_block": metadata_block(prepared),
        "timeline_section": timeline_tables(prepared),
        "glossary_section": glossary,
        "timeline_note_section": f"Timeline note: {note}" if note else "",
        "n_frames": n_frames,
        "n_questions": n_q,
        "sample_id": rec.sample_id,
        "procedure": rec.procedure,
        "procedure_category": rec.procedure_category,
        "duration": fmt_duration(prepared.probe.duration_s if prepared.probe and prepared.probe.duration_s else rec.duration_s),
    }


def build_messages(prepared: PreparedSample, cfg: Config, *, sample_dir: Optional[Path] = None,
                   log: Optional[logging.Logger] = None) -> tuple[str, list[dict]]:
    """Return ``(system_text, user_parts)`` for one prepared sample.

    ``user_parts`` = [intro text (dataset, metadata, timeline, glossary, note)] + per frame
    ``[{"type": "text", "text": caption}, {"type": "image", "path": abs path, "caption": caption}]`` +
    [final instruction text].  Frames whose file is missing are skipped with a warning.
    """
    if sample_dir is None:
        from bench.batches import prepared_dir
        sample_dir = prepared_dir(cfg, prepared.record.sample_id)
    sample_dir = Path(sample_dir)
    frames = _resolve_frames(prepared, sample_dir, log)
    mapping = _mapping(prepared, cfg, len(frames))
    system_raw, user_raw = load_prompt_texts(cfg)
    system_text = render_template(system_raw, mapping).strip()
    if FRAMES_MARKER in user_raw:
        before, after = user_raw.split(FRAMES_MARKER, 1)
    else:
        before, after = user_raw, ""
    intro = _collapse_blank_lines(render_template(before, mapping)).strip()
    outro = _collapse_blank_lines(render_template(after, mapping)).strip()
    parts: list[dict] = [{"type": "text", "text": intro}]
    n = len(frames)
    clip_phase = prepared.record.labels.get("phase") if isinstance(prepared.record.labels, dict) else None
    whole_clip = f"{clip_phase} (whole clip)" if isinstance(clip_phase, str) and not prepared.record.segments else None
    for k, (f, path) in enumerate(frames, start=1):
        label = f.label_at_t or label_at(prepared.record, f.t_s) or whole_clip
        cap = frame_caption(k, n, f.t_s, label)
        parts.append({"type": "text", "text": cap})
        parts.append({"type": "image", "path": str(path), "caption": cap, "t_s": float(f.t_s), "idx": int(f.idx)})
    if n == 0:
        parts.append({"type": "text", "text": "(No frames could be loaded for this video.)"})
    if outro:
        parts.append({"type": "text", "text": outro})
    return system_text, parts


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def frame_list(user_parts: list[dict]) -> list[dict]:
    """The image parts of ``user_parts`` as a compact, JSON-friendly list (for dry runs and logs)."""
    return [{"idx": p.get("idx"), "t_s": p.get("t_s"), "path": p.get("path"), "caption": p.get("caption")}
            for p in user_parts if p.get("type") == "image"]


def user_text(user_parts: list[dict]) -> str:
    """All text parts joined, with ``[image: <file>]`` placeholders where images go."""
    chunks: list[str] = []
    for p in user_parts:
        if p.get("type") == "text":
            chunks.append(str(p.get("text", "")))
        elif p.get("type") == "image":
            chunks.append(f"[image: {Path(str(p.get('path'))).name}]")
    return "\n\n".join(chunks)


def render_text_only(prepared: PreparedSample, cfg: Config, *, sample_dir: Optional[Path] = None) -> str:
    """Text rendering of the full prompt (system + user) for ``generate --dry-run``; never contains base64."""
    system_text, parts = build_messages(prepared, cfg, sample_dir=sample_dir)
    return f"### SYSTEM\n{system_text}\n\n### USER\n{user_text(parts)}\n"
