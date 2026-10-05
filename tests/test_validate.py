"""Offline tests for bench.validate: synthetic QASets with deliberate defects + an end-to-end run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench import validate  # noqa: E402
from bench.config import load_config  # noqa: E402
from bench.schema import QAItem, QASet  # noqa: E402

DURATION = 600.0


def make_item(qid: str, category: str, question: str, answer: str, **kw) -> QAItem:
    base = dict(
        answer_rationale="Seek to the segment, count the passes, compare with the label table.",
        evidence_timestamps=[{"start_s": 10.0, "end_s": 25.0, "observation": "capsulorhexis forceps enter"}],
        agentic_skills=["temporal_localization", "counting"],
        tool_plan=["seek 0:10", "zoom", "count"],
        answer_type="free_text",
        options=[],
        difficulty="hard",
        why_hard="needs multiple segments",
        metadata_used=["phase timeline"],
        confidence=0.85,
    )
    base.update(kw)
    return QAItem(qid=qid, category=category, question=question, answer=answer, **base)


def good_set(sample_id: str = "cataract101__case_1") -> QASet:
    return QASet(
        sample_id=sample_id,
        video_summary="Routine phaco; rhexis 0:10-1:20, phaco 2:00-5:30, IOL at 7:10.",
        questions=[
            make_item("q1", "temporal_grounding", "At what time does hydrodissection start relative to the end of the rhexis?",
                      "About 1:25, roughly 5 s after the rhexis is completed.", answer_type="timestamp"),
            make_item("q2", "complication_detection_management",
                      "Which intra-operative irregularity occurs during phacoemulsification and how is it managed?",
                      "A small anterior capsule tear is seen; the surgeon lowers flow and finishes with a chop technique.",
                      evidence_timestamps=[{"start_s": 130.0, "end_s": 200.0, "observation": "tear visible"}]),
            make_item("q3", "quantitative_estimation", "How many distinct phases are shown before lens implantation?",
                      "B. 6", answer_type="multiple_choice",
                      options=["A. 5", "B. 6", "C. 7", "D. 8"], confidence=0.7),
        ],
        generator_notes="",
        provenance={"model": "gpt-6-astra", "generated_at": "2026-10-05T10:00:00"},
    )


def codes_of(res: validate.ValidationResult, qid: str) -> set[str]:
    return {validate.issue_type(c) for c in res.question_issues.get(qid, [])}


def set_types(res: validate.ValidationResult) -> set[str]:
    return {validate.issue_type(c) for c in res.issues}


# ------------------------------------------------------------------------------------ pure checks
def test_good_set_is_ok():
    res = validate.validate_qaset(good_set(), DURATION)
    assert res.ok, res.issues
    assert not any(validate.is_hard(c) for c in res.issues)
    assert set(res.question_issues) == {"q1", "q2", "q3"}


def test_duplicate_category_flagged():
    qs = good_set()
    qs.questions[1].category = "temporal_grounding"
    res = validate.validate_qaset(qs, DURATION)
    assert not res.ok
    assert "duplicate_category" in set_types(res)
    assert "duplicate_category" in codes_of(res, "q2")
    assert "duplicate_category" not in codes_of(res, "q1")


def test_empty_answer_flagged():
    qs = good_set()
    qs.questions[0].answer = "   "
    res = validate.validate_qaset(qs, DURATION)
    assert not res.ok
    assert "empty_answer" in codes_of(res, "q1")
    assert "q1:empty_answer" in res.issues


def test_answer_contained_in_question_flagged():
    qs = good_set()
    qs.questions[0].question = "Is the capsulorhexis completed before hydrodissection at 1:25?"
    qs.questions[0].answer = "hydrodissection at 1:25"
    res = validate.validate_qaset(qs, DURATION)
    assert "answer_in_question" in codes_of(res, "q1")
    assert not res.ok


def test_boolean_answer_in_question_not_flagged():
    qs = good_set()
    qs.questions[0].question = "True or false: the rhexis is completed before hydrodissection?"
    qs.questions[0].answer = "True"
    qs.questions[0].answer_type = "boolean"
    res = validate.validate_qaset(qs, DURATION)
    assert "answer_in_question" not in codes_of(res, "q1")


def test_evidence_out_of_range_flagged():
    qs = good_set()
    qs.questions[1].evidence_timestamps = [{"start_s": 590.0, "end_s": 650.0, "observation": "beyond end"}]
    res = validate.validate_qaset(qs, DURATION)
    assert "evidence_out_of_range" in codes_of(res, "q2")
    assert not res.ok
    # within the +1 s slack is fine
    qs.questions[1].evidence_timestamps = [{"start_s": 590.0, "end_s": 600.9, "observation": "end"}]
    assert "evidence_out_of_range" not in codes_of(validate.validate_qaset(qs, DURATION), "q2")
    # negative start / end before start
    qs.questions[1].evidence_timestamps = [{"start_s": -3.0, "end_s": 5.0, "observation": "x"},
                                           {"start_s": 50.0, "end_s": 40.0, "observation": "y"}]
    res = validate.validate_qaset(qs, DURATION)
    assert {"evidence_out_of_range", "evidence_invalid"} <= codes_of(res, "q2")


def test_evidence_range_skipped_when_duration_unknown():
    qs = good_set()
    qs.questions[1].evidence_timestamps = [{"start_s": 590.0, "end_s": 5000.0, "observation": "x"}]
    res = validate.validate_qaset(qs, None)
    assert "evidence_out_of_range" not in codes_of(res, "q2")


def test_bad_multiple_choice_options():
    qs = good_set()
    qs.questions[2].options = ["A. 5", "B. 6", "C. 7"]  # only 3
    res = validate.validate_qaset(qs, DURATION)
    assert "mc_option_count" in codes_of(res, "q3")
    assert not res.ok
    qs.questions[2].options = ["A. 5", "B. 6", "C. 7", "D. 8", "E. 9", "F. 10"]  # 6
    assert "mc_option_count" in codes_of(validate.validate_qaset(qs, DURATION), "q3")


def test_multiple_choice_answer_must_reference_an_option():
    qs = good_set()
    qs.questions[2].answer = "Z. 42"
    res = validate.validate_qaset(qs, DURATION)
    assert "mc_answer_not_in_options" in codes_of(res, "q3")
    for ok_answer in ("B", "b.", "(B)", "B. 6", "6", "B) 6"):
        qs.questions[2].answer = ok_answer
        assert "mc_answer_not_in_options" not in codes_of(validate.validate_qaset(qs, DURATION), "q3"), ok_answer


def test_mc_answer_matches_helper():
    opts = ["A. Trypan blue staining", "B. Iris hooks", "C. Malyugin ring", "D. None"]
    assert validate.mc_answer_matches("C", opts)
    assert validate.mc_answer_matches("C. Malyugin ring", opts)
    assert validate.mc_answer_matches("Malyugin ring", opts)
    assert not validate.mc_answer_matches("E", opts)
    assert not validate.mc_answer_matches("Capsular tension ring", opts)


def test_options_on_non_mc_is_soft():
    qs = good_set()
    qs.questions[0].options = ["A. x", "B. y", "C. z", "D. w"]
    res = validate.validate_qaset(qs, DURATION)
    assert "options_on_non_mc" in codes_of(res, "q1")
    assert res.ok


def test_confidence_checks():
    qs = good_set()
    qs.questions[0].confidence = 1.4
    res = validate.validate_qaset(qs, DURATION)
    assert "confidence_out_of_range" in codes_of(res, "q1") and not res.ok
    qs.questions[0].confidence = 0.3
    res = validate.validate_qaset(qs, DURATION)
    assert "low_confidence" in codes_of(res, "q1")
    assert res.ok  # soft flag only
    qs.questions[0].confidence = "high"  # type: ignore[assignment]
    assert "confidence_out_of_range" in codes_of(validate.validate_qaset(qs, DURATION), "q1")


def test_duplicate_questions_by_jaccard():
    qs = good_set()
    qs.questions[1].question = "At what time does hydrodissection start relative to the end of the rhexis?"
    res = validate.validate_qaset(qs, DURATION)
    assert "duplicate_question" in set_types(res)
    assert "duplicate_question" in codes_of(res, "q2")
    assert "duplicate_question" not in codes_of(res, "q1")
    assert not res.ok


def test_question_count_and_duplicate_qid():
    qs = good_set()
    qs.questions = qs.questions[:2]
    res = validate.validate_qaset(qs, DURATION)
    assert "question_count:2" in res.issues and not res.ok
    qs = good_set()
    qs.questions[2].qid = "q1"
    res = validate.validate_qaset(qs, DURATION)
    assert "duplicate_qid:q1" in res.issues


def test_unknown_enums_flagged():
    qs = good_set()
    qs.questions[0].category = "made_up"
    qs.questions[1].answer_type = "essay"
    res = validate.validate_qaset(qs, DURATION)
    assert "unknown_category" in codes_of(res, "q1")
    assert "unknown_answer_type" in codes_of(res, "q2")
    assert not res.ok


def test_no_evidence_is_soft():
    qs = good_set()
    qs.questions[0].evidence_timestamps = []
    res = validate.validate_qaset(qs, DURATION)
    assert "no_evidence" in codes_of(res, "q1") and res.ok


def test_result_to_dict_shape():
    d = validate.validate_qaset(good_set(), DURATION).to_dict()
    assert set(d) == {"ok", "issues", "question_issues"}
    assert isinstance(d["ok"], bool) and isinstance(d["issues"], list) and isinstance(d["question_issues"], dict)


# ------------------------------------------------------------------------------------ end to end
@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("OPHBENCH_DATA_DIR", str(data))
    monkeypatch.setenv("OPHBENCH_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("OPHBENCH_UI_DIR", str(tmp_path / "ui"))
    cfg = load_config()
    cfg.ensure_dirs()
    return cfg


def _write_prepared(cfg, sample_id: str, duration: float) -> None:
    d = Path(cfg.paths.prepared) / sample_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "sample.json").write_text(json.dumps({
        "version": 1,
        "record": {"sample_id": sample_id, "dataset": sample_id.split("__")[0], "video_id": "x"},
        "probe": {"duration_s": duration, "fps": 25.0, "width": 720, "height": 540},
        "frames": [], "source_path": "source.mp4", "frames_dir": "frames", "contact_sheet": "contact_sheet.jpg",
        "preview": None, "concat_map": None, "timeline_note": "", "prepared_at": "",
    }))


def test_main_end_to_end(env):
    cfg = env
    qa_dir = Path(cfg.paths.qa)
    good = good_set("cataract101__case_1")
    bad = good_set("migs__16")
    bad.questions[1].category = "temporal_grounding"            # duplicate category
    bad.questions[2].options = ["A. 5", "B. 6"]                  # bad MC
    bad.questions[0].evidence_timestamps = [{"start_s": 0, "end_s": 999, "observation": "x"}]  # out of range (dur 120)
    for qs in (good, bad):
        (qa_dir / f"{qs.sample_id}.json").write_text(json.dumps(qs.to_dict()))
    _write_prepared(cfg, good.sample_id, 600.0)
    _write_prepared(cfg, bad.sample_id, 120.0)
    (qa_dir / "broken__x.json").write_text("{not json")                     # load error
    (qa_dir / "validation_report.json").write_text("{}")                   # must be ignored as input

    import argparse
    rc = validate.main(argparse.Namespace(command="validate"), cfg)
    assert rc == 0

    rows = [json.loads(line) for line in (qa_dir / "qa_validated.jsonl").read_text().splitlines() if line.strip()]
    assert [r["sample_id"] for r in rows] == ["cataract101__case_1", "migs__16"]
    for r in rows:
        assert set(r["validation"]) == {"ok", "issues", "question_issues"}
        assert set(r) >= {"sample_id", "video_summary", "questions", "generator_notes", "provenance", "validation"}
    by_id = {r["sample_id"]: r["validation"] for r in rows}
    assert by_id["cataract101__case_1"]["ok"] is True
    v = by_id["migs__16"]
    assert v["ok"] is False
    types = {validate.issue_type(c) for c in v["issues"]}
    assert {"duplicate_category", "mc_option_count", "evidence_out_of_range"} <= types
    assert "evidence_out_of_range" in {validate.issue_type(c) for c in v["question_issues"]["q1"]}

    report = json.loads((qa_dir / "validation_report.json").read_text())
    assert report["n_qa_sets"] == 2 and report["n_ok"] == 1 and report["n_with_issues"] == 1
    assert report["n_load_errors"] == 1 and report["issue_counts"]["load_error"] == 1
    assert report["per_dataset"]["migs"]["with_issues"] == 1
    assert report["per_dataset"]["cataract101"]["ok"] == 1
    assert report["per_category"]["temporal_grounding"]["n_questions"] == 3
    assert report["issue_counts"]["mc_option_count"] == 1
    md = (qa_dir / "validation_report.md").read_text()
    assert "# QA validation report" in md and "| migs |" in md
    runs = (Path(cfg.paths.logs) / "runs.jsonl").read_text()
    assert '"stage": "validate"' in runs


def test_main_with_no_qa_files_returns_zero(env):
    import argparse
    cfg = env
    rc = validate.main(argparse.Namespace(command="validate"), cfg)
    assert rc == 0
    assert (Path(cfg.paths.qa) / "qa_validated.jsonl").exists()
    report = json.loads((Path(cfg.paths.qa) / "validation_report.json").read_text())
    assert report["n_qa_sets"] == 0
