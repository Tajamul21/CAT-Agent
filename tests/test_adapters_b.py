"""Offline tests for the Cataract-LMM, OphNet and Ophora adapters (agent B).

They read the real dataset tables under ``cfg.paths.datasets_root`` (no video decoding, no
network) and skip when the datasets are not mounted. Expensive adapters run once per module.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from bench.config import Config, deep_merge, load_config
from bench.datasets import get_adapter
from bench.datasets.cataract_lmm import (
    PHASES, phase_display, read_zip_report, site_of, tertile_cuts, tertile_of,
)
from bench.datasets.ophnet import Clip, OphNetAdapter, best_window, n_phases_bin
from bench.datasets.ophora import (
    KeywordRule, classify_instruction, instruction_len_bin, load_keyword_rules, normalise_instruction, split_clip_id,
)
from bench.schema import PROCEDURE_CATEGORIES, VideoRecord, make_sample_id
from bench.util import parse_zip_spec

LOG = logging.getLogger("tests.adapters_b")


# ------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def cfg() -> Config:
    c = load_config()
    if not Path(c.paths.datasets_root).is_dir():
        pytest.skip(f"datasets root not mounted: {c.paths.datasets_root}")
    return c


@pytest.fixture(scope="module")
def lmm_phase(cfg) -> list[VideoRecord]:
    return get_adapter("lmm_phase", cfg, log=LOG).records()


@pytest.fixture(scope="module")
def lmm_skill(cfg) -> list[VideoRecord]:
    return get_adapter("lmm_skill", cfg, log=LOG).records()


@pytest.fixture(scope="module")
def lmm_raw(cfg) -> list[VideoRecord]:
    return get_adapter("lmm_raw", cfg, log=LOG).records()


@pytest.fixture(scope="module")
def ophnet_cases(cfg) -> list[VideoRecord]:
    return get_adapter("ophnet", cfg, log=LOG).records()


@pytest.fixture(scope="module")
def ophnet_clips(cfg) -> list[VideoRecord]:
    clip_cfg = Config(deep_merge(cfg.raw, {"sampling": {"ophnet": {"unit": "clip"}}}), root=cfg.root)
    return get_adapter("ophnet", clip_cfg, log=LOG).records()


@pytest.fixture(scope="module")
def ophora(cfg) -> list[VideoRecord]:
    return get_adapter("ophora", cfg, log=LOG).records()  # one os.scandir of ~162k files


# ------------------------------------------------------------------------------------ shared checks
def _check_common(recs: list[VideoRecord], dataset: str, id_pattern: str) -> None:
    pat = re.compile(id_pattern)
    assert recs, f"{dataset}: no records"
    assert len({r.sample_id for r in recs}) == len(recs), "sample ids must be unique"
    for r in recs:
        assert r.dataset == dataset
        assert r.sample_id == make_sample_id(dataset, r.video_id)
        assert pat.match(r.sample_id), r.sample_id
        assert r.source_kind in ("file", "zip_member", "concat_clips")
        assert r.source_paths and all(Path(p.split("!", 1)[0]).is_absolute() for p in r.source_paths)
        assert r.procedure, r.sample_id
        assert r.procedure_category in PROCEDURE_CATEGORIES
        assert r.license and r.citation
        if r.duration_s is not None:
            assert r.duration_s >= 0
        _check_segments(r)
        # strata are plain strings so they can be joined into stratum keys
        assert all(isinstance(v, str) and v for v in r.strata.values()), r.strata
        # JSON round trip through the dataclass helpers
        assert VideoRecord.from_dict(r.to_dict()).to_dict() == r.to_dict()


def _check_segments(r: VideoRecord) -> None:
    prev = -1.0
    for s in r.segments:
        assert s.start_s >= 0 and s.end_s >= s.start_s, (r.sample_id, s)
        assert s.start_s >= prev, f"{r.sample_id}: segments not sorted"
        assert s.kind in ("phase", "operation", "step", "flag")
        prev = s.start_s
    if r.segments and r.duration_s is not None:
        assert max(s.end_s for s in r.segments) <= r.duration_s + 1e-6


# ------------------------------------------------------------------------------------ lmm_phase
def test_lmm_phase_counts_and_sources(lmm_phase):
    _check_common(lmm_phase, "lmm_phase", r"^lmm_phase__PH_\d{4}_\d{4}_S\d$")
    assert len(lmm_phase) == 150
    assert all(r.source_kind == "zip_member" for r in lmm_phase)
    zips = set()
    for r in lmm_phase:
        zp, member = parse_zip_spec(r.source_paths[0])
        assert member == f"{r.video_id}.mp4"
        zips.add(zp)
    assert all(Path(z).exists() for z in zips) and len(zips) == 10


def test_lmm_phase_segments_and_labels(lmm_phase):
    for r in lmm_phase:
        assert r.segments and r.duration_s == pytest.approx(max(s.end_s for s in r.segments), abs=1e-3)
        assert r.labels["n_segments"] == len(r.segments)
        assert 0.0 <= r.labels["idle_share"] <= 1.0
        assert r.labels["site"] in ("S1", "S2") and r.labels["raw_video_id"].startswith("RV_")
        assert set(r.labels["phase_names"]) <= {label for _, label in PHASES.values()}
        assert all(s.label_id in range(1, 14) for s in r.segments)
        assert set(r.strata) == {"site", "duration_bin", "idle_share_bin"}
        assert r.fps and 20 < r.fps < 70
    sites = {r.labels["site"] for r in lmm_phase}
    assert sites == {"S1", "S2"}
    assert sum(1 for r in lmm_phase if r.labels["site"] == "S2") == 21
    # one annotation row in the dataset ends before it starts; it is repaired and noted, not dropped
    bad = [r for r in lmm_phase if any("clamped" in n for n in r.notes)]
    assert [r.video_id for r in bad] == ["PH_0037_0059_S1"]


# ------------------------------------------------------------------------------------ lmm_skill
def test_lmm_skill_counts_sources_and_strata(lmm_skill):
    _check_common(lmm_skill, "lmm_skill", r"^lmm_skill__SK_\d{4}_S\d_P\d{2}$")
    assert len(lmm_skill) == 170
    loose = [r for r in lmm_skill if r.source_kind == "file"]
    assert len(loose) == 11
    for r in loose:
        assert Path(r.source_paths[0]).exists() and "LMM-samples" in r.source_paths[0]
        assert r.labels["loose_file"] is True and r.labels["adverse_event_comment"]
        assert r.labels["manifest_adverse_event"] == r.labels["adverse_event"]
    for r in lmm_skill:
        if r.source_kind == "zip_member":
            zp, member = parse_zip_spec(r.source_paths[0])
            assert member == f"{r.video_id}.mp4" and Path(zp).exists()
        assert r.labels["tracking_annotation_zip"] and Path(r.labels["tracking_annotation_zip"]).exists()
        # rubric is 1-5 but rater-averaged indicator scores go down to 0.0 in the data
        assert all(0.0 <= r.labels[k] <= 5.0 for k in r.labels["skill_scores"])
        assert 1.0 <= r.labels["averaged"] <= 5.0
        assert r.labels["adverse_event"] in (0, 1)
        assert set(r.strata) == {"skill_tertile", "adverse_event", "site"}
    assert sum(r.labels["adverse_event"] for r in lmm_skill) == 12
    tert = {t: sum(1 for r in lmm_skill if r.strata["skill_tertile"] == t) for t in ("low", "mid", "high")}
    assert set(tert) == {"low", "mid", "high"} and min(tert.values()) >= 50


# ------------------------------------------------------------------------------------ lmm_raw
def test_lmm_raw_counts_and_phase_subset(lmm_raw):
    _check_common(lmm_raw, "lmm_raw", r"^lmm_raw__RV_\d{4}_S\d$")
    assert len(lmm_raw) == 3000
    assert all(r.source_kind == "file" and r.source_paths[0].endswith(".mp4") for r in lmm_raw)
    assert sum(1 for r in lmm_raw if r.labels["in_phase_subset"]) == 150
    assert sum(1 for r in lmm_raw if r.labels["site"] == "S2") == 70
    for r in lmm_raw:
        assert r.duration_s and r.duration_s > 0 and r.labels["frame_count"] > 0 and r.labels["file_size_mb"] > 0
        assert set(r.strata) == {"site", "duration_bin"}
        if r.labels["in_phase_subset"]:
            assert r.labels["phase_video_id"].startswith("PH_") and r.labels["phase_video_id"].endswith(r.labels["site"])
    renamed = [r for r in lmm_raw if r.labels["metadata_filename"] != r.labels["file_name"]]
    assert sorted(r.video_id for r in renamed) == ["RV_2943_S2", "RV_2955_S2"]


# ------------------------------------------------------------------------------------ ophnet
def test_ophnet_case_records(ophnet_cases):
    _check_common(ophnet_cases, "ophnet", r"^ophnet__case_\d{4}$")
    assert len(ophnet_cases) == 743
    assert all(r.source_kind == "concat_clips" for r in ophnet_cases)
    truncated = 0
    for r in ophnet_cases:
        segs = r.segments
        assert segs and segs[0].start_s == 0.0
        for a, b in zip(segs, segs[1:]):
            assert a.end_s == pytest.approx(b.start_s, abs=1e-6), f"{r.sample_id}: timeline not contiguous"
        assert segs[-1].end_s == pytest.approx(r.duration_s, abs=1e-6)
        assert len(segs) == len(r.source_paths) == r.labels["n_clips"] == len(r.labels["clips"])
        indices = [s.extra["clip_index"] for s in segs]
        assert indices == sorted(indices) and all(p.endswith(f"_{i}.mp4") for p, i in zip(r.source_paths, indices))
        assert all(s.kind == "phase" and s.label and isinstance(s.label_id, int) for s in segs)
        assert all(s.duration_s == pytest.approx(o["end_s"] - o["start_s"], abs=1e-3)
                   for s, o in zip(segs, r.labels["original_segments"]))
        assert r.labels["primary_surgery"] == r.procedure and r.labels["surgery_ids"][0] == r.labels["primary_surgery_id"]
        assert r.labels["n_phases"] == len({s.label_id for s in segs}) == len(r.labels["phase_names"])
        assert set(r.labels["phase_glossary"]) == set(r.labels["phase_names"])
        assert set(r.strata) == {"primary_surgery", "procedure_category", "duration_bin", "n_phases_bin"}
        if r.labels["truncated"]:
            truncated += 1
            assert r.duration_s <= 1200 + 1e-6 < r.labels["full_case_duration_s"]
            assert any("max_case_duration_s" in n for n in r.notes)
        else:
            assert r.duration_s == pytest.approx(r.labels["full_case_duration_s"], abs=1e-6)
    assert truncated == 10
    case2 = next(r for r in ophnet_cases if r.video_id == "case_0002")
    assert case2.labels["n_clips"] == 14 and case2.labels["surgery_ids"] == [0, 2, 12]
    assert case2.procedure_category == "cataract" and case2.segments[1].label == "Corneal Incision Creation"
    assert case2.segments[1].extra["surgery_id"] == 0
    ops = case2.labels["operations"]
    assert len(ops) == 17 and all(op["operation_name"] for op in ops)
    inside = [op for op in ops if op["concat_start_s"] is not None]
    assert inside and all(0 <= op["concat_start_s"] <= op["concat_end_s"] <= case2.duration_s + 1e-6 for op in inside)
    # challenge-file "others" bucket is named explicitly, not with the misleading label-set name
    assert any(s.label == "Others (rare phases)" for r in ophnet_cases for s in r.segments)
    assert not any(s.label == "Instrument Fabrication" for r in ophnet_cases for s in r.segments)


def test_ophnet_clip_records(ophnet_clips):
    _check_common(ophnet_clips, "ophnet", r"^ophnet__case_\d{4}_\d+$")
    assert len(ophnet_clips) == 14674
    assert all(r.source_kind == "file" and len(r.segments) == 1 and r.segments[0].start_s == 0.0 for r in ophnet_clips)
    first = next(r for r in ophnet_clips if r.video_id == "case_0002_0")
    assert first.duration_s == pytest.approx(10.0) and first.labels["case_id"] == "case_0002"
    assert set(first.strata) == {"primary_surgery", "procedure_category", "duration_bin", "n_phases_bin", "phase"}


def test_ophnet_best_window():
    durs = [100, 200, 300, 400, 500]
    phases = ["a", "b", "a", "c", "d"]
    assert best_window(durs, phases, 10_000) == (0, 4)          # everything fits
    assert best_window(durs, phases, 900) == (1, 3)              # b,a,c = 900 s, 3 phases
    assert best_window(durs, phases, 650) == (0, 2)              # a,b,a (600 s, 2 phases) beats c alone
    assert best_window(durs, phases, 50) is None                 # nothing fits
    assert best_window([], [], 100) is None
    assert n_phases_bin(1) == "1-2" and n_phases_bin(3) == "3-5" and n_phases_bin(9) == "6-9"
    assert n_phases_bin(10) == "10-14" and n_phases_bin(15) == "15+"
    clip = Clip(index=0, file=Path("/x/c_0.mp4"), start=10.0, end=20.0, phase_id=1, surgery_id=0, split="train")
    assert clip.duration == 10.0 and clip.contains(10.0, 20.0) and not clip.contains(9.0, 12.0)


def test_ophnet_window_rule_applied_by_adapter(cfg):
    small = Config(deep_merge(cfg.raw, {"sampling": {"ophnet": {"max_case_duration_s": 120}}}), root=cfg.root)
    ad = OphNetAdapter(small, log=LOG)
    rec = next(r for r in ad.iter_records() if r.video_id == "case_0002")
    assert rec.labels["truncated"] and rec.duration_s <= 120 and rec.labels["full_case_n_clips"] == 14
    assert rec.segments[0].start_s == 0.0 and rec.labels["clip_window"][0] == rec.segments[0].extra["clip_index"]


# ------------------------------------------------------------------------------------ ophora
def test_ophora_counts_and_categories(ophora):
    _check_common(ophora, "ophora", r"^ophora__[A-Za-z0-9_.-]+_\d+$")
    assert len(ophora) == 26592
    cats = {r.procedure_category for r in ophora}
    assert cats <= set(PROCEDURE_CATEGORIES) and {"cataract", "retina", "glaucoma", "cornea"} <= cats
    for r in ophora:
        assert r.source_kind == "file" and r.source_paths[0].endswith(f"/clips/{r.video_id}.mp4")
        assert r.duration_s is None and r.labels["in_filtered_28k"] is True
        assert r.labels["instruction"] and r.labels["instruction_words"] == len(r.labels["instruction"].split())
        assert f"{r.labels['source_video_id']}_{r.labels['clip_index']}" == r.video_id
        assert r.strata["instruction_len_bin"] in ("<12", "12-20", ">20")
        assert r.strata["procedure_category"] == r.procedure_category
        if r.procedure_category != "other_mixed":
            assert r.labels["category_keywords"], r.sample_id
    assert len({r.labels["source_video_id"] for r in ophora}) == 3031


def test_ophora_keyword_rules(cfg):
    rules = load_keyword_rules(Path(cfg.paths.resources) / "procedure_taxonomy.yaml", LOG)
    assert rules and rules[-1].terms == () and rules[-1].category == "other_mixed"
    assert normalise_instruction("Insert the (IOL), then...") == " insert the iol then "
    cat, label, terms = classify_instruction("Insert the IOL into the capsular bag.", rules)
    assert cat == "cataract" and label == "Cataract surgery" and " iol" in terms and "capsular bag" in terms
    assert classify_instruction("Perform a pars plana vitrectomy with ILM peel.", rules)[0] == "retina"
    assert classify_instruction("Trabeculectomy with mitomycin C application.", rules)[0] == "glaucoma"
    assert classify_instruction("Endothelial keratoplasty (DMEK) graft unfolding.", rules)[0] == "cornea"
    assert classify_instruction("Create the LASIK flap with the microkeratome.", rules)[0] == "refractive"
    assert classify_instruction("Levator advancement for ptosis repair.", rules)[0] == "oculoplastics_strabismus"
    assert classify_instruction("Close the conjunctiva with sutures.", rules) == ("other_mixed", "Other ophthalmic surgery", [])
    # retina rules come before cataract rules: a mixed caption is classified by the first matching rule
    assert classify_instruction("Vitrectomy after dropped nucleus during phaco.", rules)[0] == "retina"
    # rules with no terms are catch-alls; no rules at all -> fallback
    assert classify_instruction("anything", [KeywordRule("glaucoma", "G", ())]) == ("glaucoma", "G", [])
    assert classify_instruction("anything", [])[0] == "other_mixed"
    assert instruction_len_bin(11) == "<12" and instruction_len_bin(12) == "12-20"
    assert instruction_len_bin(20) == "12-20" and instruction_len_bin(21) == ">20"
    assert split_clip_id("2V8jWB0tU_4_4") == ("2V8jWB0tU_4", 4) and split_clip_id("abc") == ("abc", None)


# ------------------------------------------------------------------------------------ helpers
def test_lmm_helpers(cfg, tmp_path):
    assert site_of("PH_0001_2931_S2") == "S2" and site_of("SK_0130_S1_P03") == "S1"
    assert site_of("RV_0001_S1") == "S1" and site_of("nonsense") == "unknown"
    assert phase_display("IrrigationAspiration") == (6, "Irrigation-Aspiration")
    assert phase_display("Idle") == (13, "Idle") and phase_display("New") == (None, "New")
    cuts = tertile_cuts([1, 2, 3, 4, 5, 6])
    assert cuts == (3, 5)
    assert [tertile_of(v, cuts) for v in (1, 2.9, 3, 4.9, 5, 6)] == ["low", "low", "mid", "mid", "high", "high"]
    assert tertile_of(None, cuts) == "unknown"
    report = tmp_path / "ZipContentsReport.csv"
    report.write_text('﻿"ZipFileName","InternalFileName"\n"a.zip","X_1.mp4"\n"b.zip","X_2.mp4"\n', encoding="utf-8")
    assert read_zip_report(report, LOG) == {"X_1": ("a.zip", "X_1.mp4"), "X_2": ("b.zip", "X_2.mp4")}
    assert read_zip_report(tmp_path / "missing.csv", LOG) == {}
