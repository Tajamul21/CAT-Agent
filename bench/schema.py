"""Shared data types for the ophbench curation pipeline.

Every stage reads/writes these shapes (see DESIGN.md §2). All dataclasses serialise with
``to_dict()`` and rebuild with ``from_dict()``; unknown keys are ignored on load so files written
by newer code still load.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

PROMPT_VERSION = "v1"
UI_SAMPLE_VERSION = 1

DATASETS = ["cataract101", "cataract1k", "lmm_phase", "lmm_skill", "lmm_raw", "migs", "ophnet", "ophora"]

PROCEDURE_CATEGORIES = [
    "cataract", "glaucoma", "cornea", "retina", "oculoplastics_strabismus", "refractive", "other_mixed",
]

# Human-readable dataset descriptions used in prompts, reports and the annotation UI.
DATASET_META: dict[str, dict[str, str]] = {
    "cataract101": dict(
        title="Cataract-101",
        blurb="101 complete cataract (phacoemulsification) surgery videos from Klinikum Klagenfurt, "
              "4 surgeons at two experience levels, with expert annotations of 10 quasi-standardised phases.",
        license="CC BY-NC 4.0 (see LICENSE.txt); cite the MMSys 2018 paper.",
        citation="Schoeffmann K. et al. Cataract-101 - Video Dataset of 101 Cataract Surgeries. ACM MMSys 2018. doi:10.1145/3204949.3208137",
    ),
    "cataract1k": dict(
        title="Cataract-1K",
        blurb="1,000 cataract surgery videos (Klinikum Klagenfurt, 2021-2023); 303 videos carry 1-fps phase "
              "annotations (13 phases + idle, plus usage flags for trypan blue, iris hooks and Malyugin ring).",
        license="CC BY 4.0",
        citation="Ghamsarian N. et al. Cataract-1K: Cataract Surgery Dataset for Scene Segmentation, Phase Recognition, and Irregularity Detection. arXiv:2312.06295",
    ),
    "lmm_phase": dict(
        title="Cataract-LMM / Phase Recognition",
        blurb="150 complete phacoemulsification procedures from two Iranian centres (Farabi S1, Noor S2) with "
              "frame-accurate boundaries for 13 phases including Idle.",
        license="CC BY-NC-ND 4.0",
        citation="Ahmadi M.J. et al. Cataract-LMM: Large-Scale Multi-Source Multi-Task Benchmark for Deep Learning in Surgical Video Analysis. Scientific Data 2026. doi:10.1038/s41597-026-07464-0",
    ),
    "lmm_skill": dict(
        title="Cataract-LMM / Skill Assessment",
        blurb="170 capsulorhexis phase clips scored by expert surgeons on six 1-5 indicators (microscope use, "
              "instrument handling, tissue handling, motion, commencement of flap, circular completion) with an adverse-event flag.",
        license="CC BY-NC-ND 4.0",
        citation="Ahmadi M.J. et al. Cataract-LMM. Scientific Data 2026. doi:10.1038/s41597-026-07464-0",
    ),
    "lmm_raw": dict(
        title="Cataract-LMM / Raw Videos",
        blurb="3,000 de-identified, unannotated complete cataract procedures (1,134 h) from Farabi (S1) and Noor (S2) hospitals.",
        license="CC BY-NC-ND 4.0",
        citation="Ahmadi M.J. et al. Cataract-LMM. Scientific Data 2026. doi:10.1038/s41597-026-07464-0",
    ),
    "migs": dict(
        title="MIGS video dataset",
        blurb="185 minimally invasive glaucoma surgery videos (goniotomy +/- phacoemulsification, goniosynechialysis, GATT) "
              "with 8 step labels (start/end times), operation type, knife and incision count.",
        license="Research use as released by the dataset authors (see dataset page).",
        citation="MIGS video dataset (Task I temporal step annotation; Task II instrument segmentation).",
    ),
    "ophnet": dict(
        title="OphNet-2024",
        blurb="Expert-annotated ophthalmic surgery videos (cataract, glaucoma and corneal procedures) with time-localised "
              "surgical phases (and finer operations); clips here are the phase-level trimmed videos of 743 cases.",
        license="Research use (see OphNet-benchmark repository).",
        citation="Hu M. et al. OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding. ECCV 2024. arXiv:2406.07471",
    ),
    "ophora": dict(
        title="Ophora-160K",
        blurb="Short (~5 s) clips cut from narrated YouTube ophthalmic-surgery videos, each paired with an instruction-style "
              "caption; the 28K subset is filtered for subtitles/watermarks.",
        license="Apache-2.0 (YouTube-sourced content; attribution to original uploaders applies)",
        citation="Li W. et al. Ophora: A large-scale data-driven text-guided ophthalmic surgical video generation model. arXiv:2505.07449",
    ),
}

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(s: Any) -> str:
    """Keep [A-Za-z0-9_.-]; everything else becomes '_' (collapsed, trimmed)."""
    return _SAFE_RE.sub("_", str(s)).strip("_")


def make_sample_id(dataset: str, video_id: Any) -> str:
    return f"{dataset}__{safe_id(video_id)}"


def _pick(cls, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


# --------------------------------------------------------------------------------------- records
@dataclass
class Segment:
    label: str
    label_id: Optional[Any] = None
    start_s: float = 0.0
    end_s: float = 0.0
    kind: str = "phase"  # phase | operation | step | flag
    extra: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_s"] = round(self.duration_s, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(**_pick(cls, d))


@dataclass
class VideoRecord:
    sample_id: str
    dataset: str
    video_id: str
    source_kind: str = "file"  # file | zip_member | concat_clips
    source_paths: list[str] = field(default_factory=list)
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    procedure: str = ""
    procedure_category: str = "other_mixed"
    segments: list[Segment] = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    strata: dict = field(default_factory=dict)
    license: str = ""
    citation: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["segments"] = [s.to_dict() for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VideoRecord":
        kw = _pick(cls, d)
        kw["segments"] = [Segment.from_dict(s) for s in d.get("segments", []) or []]
        return cls(**kw)

    @property
    def phase_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.kind in ("phase", "step", "operation")]


@dataclass
class SampleRecord(VideoRecord):
    stratum_key: str = ""
    selection_reason: str = ""
    sample_index: int = -1

    @classmethod
    def from_video(cls, rec: VideoRecord, stratum_key: str, selection_reason: str, sample_index: int) -> "SampleRecord":
        d = rec.to_dict()
        d.update(stratum_key=stratum_key, selection_reason=selection_reason, sample_index=sample_index)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "SampleRecord":
        kw = _pick(cls, d)
        kw["segments"] = [Segment.from_dict(s) for s in d.get("segments", []) or []]
        return cls(**kw)


# --------------------------------------------------------------------------------------- media
@dataclass
class ProbeInfo:
    duration_s: float = 0.0
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    nb_frames: Optional[int] = None
    size_bytes: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeInfo":
        return cls(**_pick(cls, d))


@dataclass
class FrameInfo:
    idx: int
    t_s: float
    file: str  # path relative to the sample directory, e.g. frames/f_000_0012.5s.jpg
    label_at_t: Optional[str] = None
    reason: str = "grid"  # grid | boundary | midpoint

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FrameInfo":
        return cls(**_pick(cls, d))


@dataclass
class PreparedSample:
    record: SampleRecord
    probe: ProbeInfo
    frames: list[FrameInfo] = field(default_factory=list)
    source_path: str = "source.mp4"
    frames_dir: str = "frames"
    contact_sheet: Optional[str] = "contact_sheet.jpg"
    preview: Optional[str] = None  # preview.mp4 when generated
    concat_map: Optional[list[dict]] = None  # [{clip, offset_s, duration_s}] for concat_clips
    timeline_note: str = ""
    prepared_at: str = ""

    def to_dict(self) -> dict:
        return {
            "version": UI_SAMPLE_VERSION,
            "record": self.record.to_dict(),
            "probe": self.probe.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "source_path": self.source_path,
            "frames_dir": self.frames_dir,
            "contact_sheet": self.contact_sheet,
            "preview": self.preview,
            "concat_map": self.concat_map,
            "timeline_note": self.timeline_note,
            "prepared_at": self.prepared_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PreparedSample":
        return cls(
            record=SampleRecord.from_dict(d["record"]),
            probe=ProbeInfo.from_dict(d.get("probe", {})),
            frames=[FrameInfo.from_dict(f) for f in d.get("frames", [])],
            source_path=d.get("source_path", "source.mp4"),
            frames_dir=d.get("frames_dir", "frames"),
            contact_sheet=d.get("contact_sheet"),
            preview=d.get("preview"),
            concat_map=d.get("concat_map"),
            timeline_note=d.get("timeline_note", ""),
            prepared_at=d.get("prepared_at", ""),
        )


# --------------------------------------------------------------------------------------- QA
QA_CATEGORIES = [
    "temporal_grounding", "workflow_deviation", "complication_detection_management",
    "next_step_prediction", "skill_assessment_evidence", "instrument_anatomy_reasoning",
    "quantitative_estimation", "counterfactual_decision", "guideline_cross_reference",
    "multi_segment_comparison",
]
AGENTIC_SKILLS = [
    "temporal_localization", "counting", "measurement_estimation", "phase_recognition",
    "instrument_recognition", "anatomy_recognition", "causal_reasoning", "planning",
    "external_knowledge_retrieval", "calculation", "comparison_across_segments",
    "anomaly_detection", "decision_under_uncertainty", "verification",
]
ANSWER_TYPES = ["free_text", "timestamp", "duration", "count", "boolean", "multiple_choice", "list", "ranking"]
DIFFICULTIES = ["hard", "very_hard"]

# Strict JSON schema (OpenAI structured outputs): every property required, no extra properties,
# no minItems/maxItems/format keywords (not universally accepted in strict mode).
QA_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "video_summary": {
            "type": "string",
            "description": "2-4 sentences describing what happens in the video with approximate timestamps.",
        },
        "questions": {
            "type": "array",
            "description": "Exactly three questions with distinct categories.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "qid": {"type": "string", "enum": ["q1", "q2", "q3"]},
                    "category": {"type": "string", "enum": QA_CATEGORIES},
                    "question": {"type": "string"},
                    "answer": {"type": "string", "description": "Gold answer, concise and verifiable."},
                    "answer_rationale": {"type": "string", "description": "Step-by-step reasoning with timestamps."},
                    "evidence_timestamps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "start_s": {"type": "number"},
                                "end_s": {"type": "number"},
                                "observation": {"type": "string"},
                            },
                            "required": ["start_s", "end_s", "observation"],
                        },
                    },
                    "agentic_skills": {"type": "array", "items": {"type": "string", "enum": AGENTIC_SKILLS}},
                    "tool_plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered steps an agent would execute (seek, zoom, count, measure, look up).",
                    },
                    "answer_type": {"type": "string", "enum": ANSWER_TYPES},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Options 'A. ...' for multiple_choice; empty list otherwise.",
                    },
                    "difficulty": {"type": "string", "enum": DIFFICULTIES},
                    "why_hard": {"type": "string"},
                    "metadata_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of provided label fields used to write/verify the answer.",
                    },
                    "confidence": {"type": "number", "description": "0-1 confidence that the gold answer is correct."},
                },
                "required": [
                    "qid", "category", "question", "answer", "answer_rationale", "evidence_timestamps",
                    "agentic_skills", "tool_plan", "answer_type", "options", "difficulty", "why_hard",
                    "metadata_used", "confidence",
                ],
            },
        },
        "generator_notes": {"type": "string", "description": "Caveats, uncertainty, or data issues noticed."},
    },
    "required": ["video_summary", "questions", "generator_notes"],
}


@dataclass
class QAItem:
    qid: str
    category: str
    question: str
    answer: str
    answer_rationale: str = ""
    evidence_timestamps: list[dict] = field(default_factory=list)
    agentic_skills: list[str] = field(default_factory=list)
    tool_plan: list[str] = field(default_factory=list)
    answer_type: str = "free_text"
    options: list[str] = field(default_factory=list)
    difficulty: str = "hard"
    why_hard: str = ""
    metadata_used: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QAItem":
        kw = _pick(cls, d)
        kw.setdefault("qid", "q?")
        kw.setdefault("category", "temporal_grounding")
        kw.setdefault("question", "")
        kw.setdefault("answer", "")
        if kw.get("options") is None:
            kw["options"] = []
        return cls(**kw)


@dataclass
class QASet:
    sample_id: str
    video_summary: str = ""
    questions: list[QAItem] = field(default_factory=list)
    generator_notes: str = ""
    provenance: dict = field(default_factory=dict)  # model, route, effort, n_frames, usage, latency_s, generated_at, prompt_version

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "video_summary": self.video_summary,
            "questions": [q.to_dict() for q in self.questions],
            "generator_notes": self.generator_notes,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QASet":
        return cls(
            sample_id=d["sample_id"],
            video_summary=d.get("video_summary", ""),
            questions=[QAItem.from_dict(q) for q in d.get("questions", [])],
            generator_notes=d.get("generator_notes", ""),
            provenance=d.get("provenance", {}),
        )

    @classmethod
    def from_llm_json(cls, sample_id: str, data: dict, provenance: dict) -> "QASet":
        """Build from the model's JSON (already parsed). Tolerant to minor deviations."""
        qs = []
        for i, q in enumerate(data.get("questions", []) or []):
            if not isinstance(q, dict):
                continue
            q = dict(q)
            q.setdefault("qid", f"q{i + 1}")
            qs.append(QAItem.from_dict(q))
        return cls(
            sample_id=sample_id,
            video_summary=str(data.get("video_summary", "")),
            questions=qs,
            generator_notes=str(data.get("generator_notes", "")),
            provenance=provenance,
        )


@dataclass
class GenerationLogEntry:
    sample_id: str
    model: str
    route: str
    reasoning_effort: str
    n_frames: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    latency_s: Optional[float] = None
    status: str = "ok"  # ok | parse_error | error | dry_run
    error: Optional[str] = None
    request_id: Optional[str] = None
    est_cost_usd: Optional[float] = None
    started_at: str = ""
    finished_at: str = ""
    attempts: int = 1

    def to_dict(self) -> dict:
        return asdict(self)
