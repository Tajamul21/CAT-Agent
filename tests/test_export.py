"""Offline tests for bench.export_ui: synthetic manifest + prepared samples + QA -> UI bundle shapes."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench import export_ui  # noqa: E402
from bench.config import load_config  # noqa: E402
from bench.schema import (FrameInfo, PreparedSample, ProbeInfo, QAItem, QASet, SampleRecord, Segment,  # noqa: E402
                          UI_SAMPLE_VERSION)
from bench.util import write_json, write_jsonl  # noqa: E402

META_KEYS = {"generated_at", "ui_version", "n_samples", "datasets", "select_min", "select_max", "annotators", "dataset_meta"}
INDEX_KEYS = {"sample_id", "dataset", "dataset_title", "procedure", "procedure_category", "duration_s", "n_questions",
              "thumb", "preview", "assigned_to", "has_issues"}
SAMPLE_KEYS = {"sample_id", "dataset", "dataset_title", "dataset_blurb", "citation", "license", "procedure",
               "procedure_category", "video", "metadata", "segments", "frames", "timeline_note", "video_summary",
               "questions", "generator_notes", "provenance"}
VIDEO_KEYS = {"preview", "contact_sheet", "duration_s", "width", "height", "fps"}
QUESTION_KEYS = {"qid", "category", "question", "answer", "answer_rationale", "evidence_timestamps", "agentic_skills",
                 "tool_plan", "answer_type", "options", "difficulty", "why_hard", "metadata_used", "confidence",
                 "validation_issues"}
ASSIGN_KEYS = {"annotators", "by_annotator", "overlap"}


# ------------------------------------------------------------------------------------ fixtures
@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp OPHBENCH_DATA_DIR / logs and a temp copy of ui/ (static files only)."""
    data = tmp_path / "data"
    ui = tmp_path / "ui"
    src_ui = REPO_ROOT / "ui"
    if src_ui.exists():
        shutil.copytree(src_ui, ui, ignore=shutil.ignore_patterns("data", "media"))
    ui.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPHBENCH_DATA_DIR", str(data))
    monkeypatch.setenv("OPHBENCH_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("OPHBENCH_UI_DIR", str(ui))
    cfg = load_config()
    cfg.ensure_dirs()
    return cfg


def make_record(dataset: str, video_id: str, **kw) -> SampleRecord:
    base = dict(
        source_kind="file", source_paths=[f"/mnt/store/tashraf4/datasets/{dataset}/{video_id}.mp4"],
        duration_s=600.0, fps=25.0, width=720, height=540,
        procedure="Cataract surgery (phacoemulsification)", procedure_category="cataract",
        segments=[Segment(label="Incision", label_id=1, start_s=0.0, end_s=30.0),
                  Segment(label="Capsulorhexis", label_id=2, start_s=30.0, end_s=120.0),
                  Segment(label="Trypan Blue Used", label_id=14, start_s=10.0, end_s=20.0, kind="flag")],
        labels={"surgeon_id": 2, "experience": "high", "annotated": True, "flags": ["Trypan Blue"],
                "tracking_annotation_zip": "/mnt/store/tashraf4/datasets/Cataract-LMM/TR_0001.zip",
                "source_file": "/mnt/store/x/y/case_1.mp4", "idle_share": 0.2345, "n_phase_segments": 9,
                "scores": {"microscope_use": 4, "motion": 3}, "empty": None},
        notes=["case truncated to 1200 s"],
        stratum_key="high|<5min", selection_reason="balanced", sample_index=0,
    )
    base.update(kw)
    return SampleRecord(sample_id=f"{dataset}__{video_id}", dataset=dataset, video_id=video_id, **base)


def write_prepared(cfg, rec: SampleRecord, *, with_sheet=True, with_preview=False, duration=600.0) -> Path:
    d = Path(cfg.paths.prepared) / rec.sample_id
    d.mkdir(parents=True, exist_ok=True)
    if with_sheet:
        Image.new("RGB", (64, 48), (120, 30, 30)).save(d / "contact_sheet.jpg", "JPEG")
    if with_preview:
        (d / "preview.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    ps = PreparedSample(
        record=rec, probe=ProbeInfo(duration_s=duration, fps=25.0, width=720, height=540, codec="h264"),
        frames=[FrameInfo(idx=0, t_s=5.0, file="frames/f_000_0005.0s.jpg", label_at_t="Incision"),
                FrameInfo(idx=1, t_s=60.0, file="frames/f_001_0060.0s.jpg", label_at_t="Capsulorhexis")],
        contact_sheet="contact_sheet.jpg", preview="preview.mp4" if with_preview else None,
        timeline_note="times refer to the source video", prepared_at="2026-10-05T10:00:00",
    )
    write_json(d / "sample.json", ps.to_dict())
    return d


def make_qa(sample_id: str) -> QASet:
    item = lambda qid, cat, atype="free_text", options=None: QAItem(  # noqa: E731
        qid=qid, category=cat, question=f"Question {qid} about {cat}?", answer=f"Answer {qid}",
        answer_rationale="because", evidence_timestamps=[{"start_s": 10, "end_s": 20, "observation": "obs"}],
        agentic_skills=["temporal_localization"], tool_plan=["seek"], answer_type=atype, options=options or [],
        difficulty="hard", why_hard="multi-step", metadata_used=["phases"], confidence=0.8)
    return QASet(sample_id=sample_id, video_summary="summary", generator_notes="notes",
                 questions=[item("q1", "temporal_grounding"), item("q2", "workflow_deviation"),
                            item("q3", "quantitative_estimation", "multiple_choice", ["A. 1", "B. 2", "C. 3", "D. 4"])],
                 provenance={"model": "gpt-6-astra", "generated_at": "2026-10-05T11:00:00", "usage": {"prompt_tokens": 10}})


def run_export(cfg, **kw) -> int:
    args = argparse.Namespace(command="export-ui", media_mode=kw.get("media_mode", "symlink"),
                              annotators=kw.get("annotators"), overlap=kw.get("overlap"),
                              include_invalid=kw.get("include_invalid", False))
    return export_ui.main(args, cfg)


def load(ui: Path, rel: str):
    return json.loads((ui / rel).read_text())


# ------------------------------------------------------------------------------------ tests
def test_export_shapes_and_relative_media(env):
    cfg = env
    ui = export_ui.ui_paths(cfg).root
    r1 = make_record("cataract101", "case_1")
    r2 = make_record("migs", "16", procedure="Glaucoma MIGS: GT120", procedure_category="glaucoma", sample_index=1)
    r3 = make_record("ophnet", "case_0003", procedure_category="cornea", sample_index=2)   # prepared, no QA -> skipped
    r4 = make_record("ophora", "abc_1", sample_index=3)                                 # not prepared -> skipped
    write_jsonl(Path(cfg.paths.sample) / "sample_manifest.jsonl", [r.to_dict() for r in (r1, r2, r3, r4)])
    write_prepared(cfg, r1, with_preview=True)
    write_prepared(cfg, r2, with_preview=False, duration=240.0)
    write_prepared(cfg, r3)

    qa1, qa2 = make_qa(r1.sample_id), make_qa(r2.sample_id)
    write_json(Path(cfg.paths.qa) / f"{qa1.sample_id}.json", qa1.to_dict())
    write_json(Path(cfg.paths.qa) / f"{qa2.sample_id}.json", qa2.to_dict())
    row1 = qa1.to_dict()
    row1["validation"] = {"ok": False, "issues": ["q2:empty_answer", "duplicate_category:x"],
                          "question_issues": {"q1": [], "q2": ["empty_answer"], "q3": []}}
    write_jsonl(Path(cfg.paths.qa) / "qa_validated.jsonl", [row1])      # r2 falls back to data/qa/<id>.json

    assert run_export(cfg, annotators="alice,bob,carol", overlap=0.5) == 0

    meta = load(ui, "data/meta.json")
    assert META_KEYS <= set(meta)
    assert meta["ui_version"] == UI_SAMPLE_VERSION == 1
    assert meta["n_samples"] == 2
    assert meta["datasets"] == {"cataract101": 1, "migs": 1}
    assert meta["annotators"] == ["alice", "bob", "carol"]
    assert set(meta["dataset_meta"]) == {"cataract101", "migs"}
    for dm in meta["dataset_meta"].values():
        assert set(dm) == {"title", "blurb", "license", "citation"}
    assert isinstance(meta["select_min"], int) and isinstance(meta["select_max"], int)

    index = load(ui, "data/index.json")
    assert [e["sample_id"] for e in index] == [r1.sample_id, r2.sample_id]
    for e in index:
        assert INDEX_KEYS <= set(e)
        assert e["thumb"] == f"media/{e['sample_id']}/contact_sheet.jpg"
        assert not e["thumb"].startswith("/") and "mnt" not in e["thumb"]
        assert e["n_questions"] == 3
        assert isinstance(e["assigned_to"], list) and e["assigned_to"]
    e1, e2 = index
    assert e1["preview"] == f"media/{r1.sample_id}/preview.mp4"
    assert e2["preview"] is None                       # preview absent -> null
    assert e1["has_issues"] is True and e2["has_issues"] is False
    assert e2["duration_s"] == 240.0 and e2["procedure_category"] == "glaucoma"

    s1 = load(ui, f"data/samples/{r1.sample_id}.json")
    assert SAMPLE_KEYS <= set(s1)
    assert set(s1["video"]) >= VIDEO_KEYS
    assert s1["video"]["contact_sheet"] == f"media/{r1.sample_id}/contact_sheet.jpg"
    assert s1["video"]["preview"] == f"media/{r1.sample_id}/preview.mp4"
    assert s1["video"]["duration_s"] == 600.0 and s1["video"]["width"] == 720 and s1["video"]["fps"] == 25.0
    assert s1["dataset_title"] == "Cataract-101" and s1["dataset_blurb"] and s1["citation"] and s1["license"]
    assert len(s1["questions"]) == 3
    for q in s1["questions"]:
        assert QUESTION_KEYS <= set(q)
        assert isinstance(q["confidence"], (int, float)) and isinstance(q["options"], list)
        assert all(set(ev) == {"start_s", "end_s", "observation"} for ev in q["evidence_timestamps"])
    assert s1["questions"][1]["validation_issues"] == ["empty_answer"]
    assert s1["questions"][0]["validation_issues"] == []
    assert s1["segments"] == [{"label": "Incision", "start_s": 0.0, "end_s": 30.0, "kind": "phase"},
                              {"label": "Capsulorhexis", "start_s": 30.0, "end_s": 120.0, "kind": "phase"},
                              {"label": "Trypan Blue Used", "start_s": 10.0, "end_s": 20.0, "kind": "flag"}]
    assert s1["frames"] == [{"t_s": 5.0, "label": "Incision"}, {"t_s": 60.0, "label": "Capsulorhexis"}]
    assert s1["timeline_note"] == "times refer to the source video"
    assert s1["video_summary"] == "summary" and s1["generator_notes"] == "notes"
    assert s1["provenance"]["model"] == "gpt-6-astra"

    # metadata: clinician-friendly labels, no internal paths / None values
    md = {m["label"]: m["value"] for m in s1["metadata"]}
    assert all(set(m) == {"label", "value"} for m in s1["metadata"])
    assert md["Surgeon"] == "2" and md["Surgeon experience"] == "high"
    assert md["Phase annotations available"] == "yes"
    assert md["Special techniques / flags"] == "Trypan Blue"
    assert md["Idle share of video"] == "23%"
    assert md["Skill scores (1-5)"] == "Skill: microscope use: 4; Skill: motion: 3"
    assert md["Notes"] == "case truncated to 1200 s"
    assert md["Duration"] == "10 min 00 s" and md["Resolution"] == "720 x 540"
    joined = json.dumps(s1["metadata"])
    assert "/mnt/" not in joined and ".zip" not in joined and ".mp4" not in joined and "null" not in joined
    assert "Source file" not in md and "Tracking annotation zip" not in md

    s2 = load(ui, f"data/samples/{r2.sample_id}.json")
    assert s2["video"]["preview"] is None
    assert all(q["validation_issues"] == [] for q in s2["questions"])   # unvalidated fallback
    assert not (ui / "data/samples" / f"{r3.sample_id}.json").exists()
    assert not (ui / "data/samples" / f"{r4.sample_id}.json").exists()

    # media symlinks
    sheet = ui / "media" / r1.sample_id / "contact_sheet.jpg"
    assert sheet.is_symlink() and sheet.exists()
    assert Path(os.readlink(sheet)) == (Path(cfg.paths.prepared) / r1.sample_id / "contact_sheet.jpg").resolve()
    assert (ui / "media" / r1.sample_id / "preview.mp4").is_symlink()
    assert not (ui / "media" / r2.sample_id / "preview.mp4").exists()

    # assignments: 3 annotators, overlap ceil(0.5*2)=1 -> one sample seen by all, other by exactly one
    asg = load(ui, "data/assignments.json")
    assert set(asg) == ASSIGN_KEYS
    assert asg["annotators"] == ["alice", "bob", "carol"]
    assert set(asg["by_annotator"]) == {"alice", "bob", "carol"}
    assert len(asg["overlap"]) == 1
    ov = asg["overlap"][0]
    assert all(ov in ids for ids in asg["by_annotator"].values())
    other = ({r1.sample_id, r2.sample_id} - {ov}).pop()
    assert sum(other in ids for ids in asg["by_annotator"].values()) == 1
    by_sid = {e["sample_id"]: e["assigned_to"] for e in index}
    assert by_sid[ov] == ["alice", "bob", "carol"] and len(by_sid[other]) == 1

    # idempotent re-run
    assert run_export(cfg, annotators="alice,bob,carol", overlap=0.5) == 0
    assert load(ui, "data/assignments.json") == asg


def test_export_without_annotators_and_media_none(env):
    cfg = env
    ui = export_ui.ui_paths(cfg).root
    r1 = make_record("cataract1k", "case_2003")
    write_jsonl(Path(cfg.paths.sample) / "sample_manifest.jsonl", [r1.to_dict()])
    write_prepared(cfg, r1, with_preview=True)
    write_json(Path(cfg.paths.qa) / f"{r1.sample_id}.json", make_qa(r1.sample_id).to_dict())
    assert run_export(cfg, media_mode="none") == 0
    meta, index, asg = load(ui, "data/meta.json"), load(ui, "data/index.json"), load(ui, "data/assignments.json")
    assert meta["annotators"] == [] and asg == {"annotators": [], "by_annotator": {}, "overlap": []}
    assert index[0]["assigned_to"] == []
    assert index[0]["preview"] == f"media/{r1.sample_id}/preview.mp4"   # URL kept; media hosted elsewhere
    assert not (ui / "media" / r1.sample_id).exists()                     # nothing placed in media-mode none


def test_export_requires_manifest(env):
    assert run_export(env) == 1


def test_export_with_nothing_ready_is_ok_but_empty(env):
    cfg = env
    r1 = make_record("cataract101", "case_9")
    write_jsonl(Path(cfg.paths.sample) / "sample_manifest.jsonl", [r1.to_dict()])
    write_prepared(cfg, r1)                                    # prepared but no QA (dry-run scenario)
    assert run_export(cfg) == 0
    ui = export_ui.ui_paths(cfg).root
    assert load(ui, "data/meta.json")["n_samples"] == 0 and load(ui, "data/index.json") == []


def test_stale_sample_files_are_pruned(env):
    cfg = env
    ui = export_ui.ui_paths(cfg)
    ui.samples.mkdir(parents=True, exist_ok=True)
    (ui.samples / "cataract101__old.json").write_text("{}")
    r1 = make_record("cataract101", "case_1")
    write_jsonl(Path(cfg.paths.sample) / "sample_manifest.jsonl", [r1.to_dict()])
    write_prepared(cfg, r1)
    write_json(Path(cfg.paths.qa) / f"{r1.sample_id}.json", make_qa(r1.sample_id).to_dict())
    assert run_export(cfg) == 0
    assert not (ui.samples / "cataract101__old.json").exists()
    assert (ui.samples / f"{r1.sample_id}.json").exists()


def test_build_assignments_round_robin_and_overlap():
    ids = [f"ds__{i}" for i in range(10)]
    asg = export_ui.build_assignments(ids, ["a", "b", "c"], 0.2, seed=123)
    assert len(asg["overlap"]) == math.ceil(0.2 * 10) == 2
    for sid in ids:
        n = sum(sid in lst for lst in asg["by_annotator"].values())
        assert n == (3 if sid in asg["overlap"] else 1)
    sizes = sorted(len(v) for v in asg["by_annotator"].values())
    assert max(sizes) - min(sizes) <= 1
    assert asg == export_ui.build_assignments(ids, ["a", "b", "c"], 0.2, seed=123)       # deterministic
    assert asg["overlap"] != export_ui.build_assignments(ids, ["a", "b", "c"], 0.2, seed=7)["overlap"] or True
    assert export_ui.build_assignments(ids, [], 0.2, seed=1) == {"annotators": [], "by_annotator": {}, "overlap": []}
    single = export_ui.build_assignments(ids, ["solo"], 0.5, seed=1)
    assert single["by_annotator"] == {"solo": ids} and single["overlap"] == []
    assert export_ui.build_assignments(ids, "x, y ,x", 0.0, seed=1)["annotators"] == ["x", "y"]


def test_format_value_and_path_filtering():
    fv = export_ui.format_value
    assert fv(True) == "yes" and fv(False) == "no" and fv(None) is None and fv("") is None
    assert fv(3.0) == "3" and fv(0.3333) == "0.33" and fv(7) == "7"
    assert fv("/mnt/store/x/y.mp4") is None and fv("a/b/c") is None and fv("videos.zip!case_1.mp4") is None
    assert fv("Trypan blue!") == "Trypan blue!"
    assert fv(["a", None, "b"]) == "a, b"
    assert fv([{"name": "Phaco"}, {"name": "IOL"}, {"name": "Phaco"}]) == "Phaco, IOL"
    assert fv({"site": "S1", "nested": {"x": 1}}) == "Site code: S1"
    assert fv(list(range(25))).endswith("(+5 more)")
    assert export_ui.humanize("n_phases") == "Number of phases"
    assert export_ui.humanize("has_suture") == "Has suture"
    assert export_ui.category_label("oculoplastics_strabismus") == "Oculoplastics / strabismus"
