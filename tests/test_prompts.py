"""Offline tests for bench.prompts (prompt builder)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from bench import prompts  # noqa: E402
from bench.config import load_config  # noqa: E402
from bench.schema import FrameInfo, PreparedSample, ProbeInfo, SampleRecord, Segment  # noqa: E402
from bench.util import fmt_time  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _make_frames(sample_dir: Path, times: list[float], labels: list[str | None]) -> list[FrameInfo]:
    fdir = sample_dir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (t, lab) in enumerate(zip(times, labels)):
        name = f"frames/f_{i:03d}_{t:07.1f}s.jpg"
        Image.new("RGB", (64, 48), color=(i * 40 % 255, 80, 120)).save(sample_dir / name, "JPEG", quality=80)
        frames.append(FrameInfo(idx=i, t_s=t, file=name, label_at_t=lab))
    return frames


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[PreparedSample, Path]:
    sample_dir = tmp_path / "cataract101__case_999"
    record = SampleRecord(
        sample_id="cataract101__case_999", dataset="cataract101", video_id="case_999", source_kind="file",
        source_paths=["/nonexistent/case_999.mp4"], duration_s=30.0, fps=25.0, width=720, height=540,
        procedure="Cataract surgery (phacoemulsification)", procedure_category="cataract",
        segments=[
            Segment(label="Incision", label_id=1, start_s=0.0, end_s=10.0),
            Segment(label="Capsulorhexis", label_id=2, start_s=10.0, end_s=22.5),
            Segment(label="Hydrodissection", label_id=3, start_s=22.5, end_s=30.0),
            Segment(label="Trypan Blue Used", label_id=14, start_s=5.0, end_s=6.0, kind="flag"),
        ],
        labels={"surgeon_id": 2, "experience": "high", "frames": 750, "fps": 25, "n_phase_segments": 3,
                "flags": ["Trypan Blue"], "skill_scores": {"motion": 4, "tissue_handling": 3}},
        strata={"experience": "high"}, license="Open access for research", citation="Schoeffmann 2018",
        stratum_key="experience=high", selection_reason="balanced", sample_index=0,
    )
    frames = _make_frames(sample_dir, [5.0, 15.0, 26.0], ["Incision", None, "Hydrodissection"])
    ps = PreparedSample(record=record, probe=ProbeInfo(duration_s=30.0, fps=25.0, width=720, height=540, codec="h264"),
                        frames=frames, timeline_note="times refer to the original video")
    return ps, sample_dir


def test_frame_caption_format():
    assert prompts.frame_caption(1, 3, 5.0, "Incision") == "Frame 1/3 - t=00:05.0 - Incision"
    assert prompts.frame_caption(12, 48, 3725.25, None) == f"Frame 12/48 - t={fmt_time(3725.25)} - no phase label"


def test_label_at_prefers_phase_segments(prepared):
    ps, _ = prepared
    assert prompts.label_at(ps.record, 15.0) == "Capsulorhexis"
    assert prompts.label_at(ps.record, 5.5) == "Incision"  # the flag segment is not a phase
    assert prompts.label_at(ps.record, 99.0) is None


def test_build_messages_structure(prepared, cfg):
    ps, sample_dir = prepared
    system_text, parts = build = prompts.build_messages(ps, cfg, sample_dir=sample_dir)
    assert "agentic" in system_text.lower() and "json" in system_text.lower()
    assert "exactly 3" in system_text.lower() or "exactly three" in system_text.lower()
    assert all(p["type"] in ("text", "image") for p in parts)
    images = [p for p in parts if p["type"] == "image"]
    assert len(images) == 3
    for img in images:
        assert Path(img["path"]).is_absolute() and Path(img["path"]).exists()
        assert img["caption"].startswith("Frame ")
    # every image is immediately preceded by its caption text part
    for i, p in enumerate(parts):
        if p["type"] == "image":
            assert parts[i - 1] == {"type": "text", "text": p["caption"]}
    captions = [p["caption"] for p in images]
    assert captions[0] == "Frame 1/3 - t=00:05.0 - Incision"
    assert captions[1] == "Frame 2/3 - t=00:15.0 - Capsulorhexis"  # label_at_t None -> derived from segments
    assert captions[2] == "Frame 3/3 - t=00:26.0 - Hydrodissection"
    intro, outro = parts[0]["text"], parts[-1]["text"]
    assert parts[0]["type"] == "text" and parts[-1]["type"] == "text"
    assert "Cataract-101" in intro and "Procedure: Cataract surgery" in intro
    assert "Surgeon experience level: high" in intro and "Surgeon id (anonymised): 2" in intro
    assert "Expert skill scores" in intro and "motion: 4" in intro
    assert "times refer to the original video" in intro
    assert "3 sampled frames" in intro
    assert "frames" not in intro.split("## Frames")[0].split("## Video metadata")[1].lower().replace("frames concatenated", "")  # bulky label skipped
    assert "Return only the JSON" in outro and "exactly 3" in outro


def test_timeline_table(prepared, cfg):
    ps, sample_dir = prepared
    table = prompts.timeline_tables(ps)
    assert "| name | start | end | duration |" in table
    assert "| Incision | 00:00.0 | 00:10.0 | 00:10.0 |" in table
    assert "| Capsulorhexis | 00:10.0 | 00:22.5 | 00:12.5 |" in table
    assert "**Phases** (3)" in table and "**Usage flags** (1)" in table
    assert "| Trypan Blue Used | 00:05.0 | 00:06.0 | 00:01.0 |" in table
    _, parts = prompts.build_messages(ps, cfg, sample_dir=sample_dir)
    assert "| Incision | 00:00.0 | 00:10.0 | 00:10.0 |" in parts[0]["text"]


def test_no_segments_message(prepared, cfg):
    ps, sample_dir = prepared
    ps.record.segments = []
    _, parts = prompts.build_messages(ps, cfg, sample_dir=sample_dir)
    assert "No expert label timeline" in parts[0]["text"]


def test_missing_frame_is_skipped(prepared, cfg):
    ps, sample_dir = prepared
    (sample_dir / ps.frames[1].file).unlink()
    _, parts = prompts.build_messages(ps, cfg, sample_dir=sample_dir)
    images = [p for p in parts if p["type"] == "image"]
    assert [p["caption"] for p in images] == ["Frame 1/2 - t=00:05.0 - Incision", "Frame 2/2 - t=00:26.0 - Hydrodissection"]
    assert "2 sampled frames" in parts[0]["text"]


def test_render_text_only_has_no_base64(prepared, cfg):
    ps, sample_dir = prepared
    text = prompts.render_text_only(ps, cfg, sample_dir=sample_dir)
    assert "### SYSTEM" in text and "### USER" in text
    assert text.count("[image: f_") == 3
    assert "base64" not in text


def test_ophnet_glossary_from_resources(cfg, tmp_path):
    sample_dir = tmp_path / "ophnet__case_0002"
    record = SampleRecord(sample_id="ophnet__case_0002", dataset="ophnet", video_id="case_0002", source_kind="concat_clips",
                          duration_s=40.0, procedure="Phacoemulsification", procedure_category="cataract",
                          segments=[Segment(label="Step Interval", label_id=2, start_s=0.0, end_s=20.0),
                                    Segment(label="Non-functional Segment", label_id=3, start_s=20.0, end_s=40.0)],
                          labels={"n_clips": 2, "operations": [{"name": "Withdrawal of Puncture Knife"}, {"name": "Step Interval"}]})
    frames = _make_frames(sample_dir, [10.0, 30.0], [None, None])
    ps = PreparedSample(record=record, probe=ProbeInfo(duration_s=40.0), frames=frames,
                        concat_map=[{"clip": "a", "offset_s": 0, "duration_s": 20}, {"clip": "b", "offset_s": 20, "duration_s": 20}],
                        timeline_note="times refer to the concatenated case video")
    glossary = prompts.phase_glossary(ps, Path(cfg.paths.resources))
    assert glossary.startswith("### Label glossary")
    assert "Step Interval: This is a surgical screen" in glossary
    _, parts = prompts.build_messages(ps, cfg, sample_dir=sample_dir)
    intro = parts[0]["text"]
    assert "2 phase clips concatenated" in intro
    assert "Finer operations annotated" in intro and "Withdrawal of Puncture Knife" in intro
    assert "OphNet-2024" in intro


def test_render_template_leaves_unknown_braces():
    out = prompts.render_template("a {known} b {unknown} {{literal}}", {"known": 1})
    assert out == "a 1 b {unknown} {{literal}}"


def test_prompt_fingerprint_stable(cfg):
    assert prompts.prompt_fingerprint(cfg) == prompts.prompt_fingerprint(cfg)
    assert len(prompts.prompt_fingerprint(cfg)) == 12
    assert prompts.prompt_version(cfg) == "v1"


def test_whole_clip_phase_label_when_no_segments(cfg, tmp_path):
    sample_dir = tmp_path / "lmm_skill__SK_0001_S1_P03"
    record = SampleRecord(sample_id="lmm_skill__SK_0001_S1_P03", dataset="lmm_skill", video_id="SK_0001_S1_P03",
                          procedure="Cataract surgery - capsulorhexis phase clip", procedure_category="cataract",
                          labels={"phase": "Capsulorhexis", "adverse_event": 1, "skill_scores": {"motion": 3.88, "circular_completion": 4.64},
                                  "motion": 3.88, "circular_completion": 4.64, "tracking_annotation_zip": "/mnt/x/TR_0001.zip"})
    frames = _make_frames(sample_dir, [2.8, 30.0], [None, None])
    ps = PreparedSample(record=record, probe=ProbeInfo(duration_s=68.3, width=720, height=480, fps=29.97), frames=frames)
    _, parts = prompts.build_messages(ps, cfg, sample_dir=sample_dir)
    captions = [p["caption"] for p in parts if p["type"] == "image"]
    assert captions == ["Frame 1/2 - t=00:02.8 - Capsulorhexis (whole clip)", "Frame 2/2 - t=00:30.0 - Capsulorhexis (whole clip)"]
    intro = parts[0]["text"]
    assert "Adverse event flagged by annotators: yes" in intro
    assert "Expert skill scores (1-5, higher is better): motion: 3.88; circular completion: 4.64" in intro
    assert intro.count("3.88") == 1  # individual indicator keys hidden when the dict is present
    assert "/mnt/x/TR_0001.zip" not in intro  # file paths never reach the prompt
