"""Offline tests for the agent-A dataset adapters: cataract101, cataract1k, migs (DESIGN.md §3.1/3.2/3.4).

The real-data tests read only metadata (CSV/JSON files and zip tables of contents), never video, and run
in a few seconds. Synthetic fixtures cover the cataract1k usage flags (which do not occur in the on-disk
copy of annotations.csv) and the MIGS ``' ' -> '_'`` file mapping. Tests skip when a dataset is absent.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench.config import DEFAULTS, Config, deep_merge, load_config  # noqa: E402
from bench.datasets import available_adapters, get_adapter  # noqa: E402
from bench.datasets.cataract1k import SIZE_BIN_LABELS, Cataract1kAdapter, size_bin, size_quartile_cuts  # noqa: E402
from bench.datasets.cataract101 import LEAD_LABEL  # noqa: E402
from bench.datasets.migs import (  # noqa: E402
    MigsAdapter, normalise_knife, normalise_label, normalise_video_key, parse_operation_type, resolve_migs_files,
)
from bench.schema import PROCEDURE_CATEGORIES, VideoRecord, make_sample_id  # noqa: E402

CFG = load_config(root=REPO)
DS = Path(CFG.paths.datasets_root)
SAMPLE_ID_RE = re.compile(r"^[a-z0-9_]+__[A-Za-z0-9_.-]+$")

needs_c101 = pytest.mark.skipif(not (DS / "cataract-101" / "videos.csv").exists(), reason="cataract-101 not available")
needs_c1k = pytest.mark.skipif(not (DS / "cataract-1k" / "cat-1k").is_dir(), reason="cataract-1k not available")
needs_migs = pytest.mark.skipif(not (DS / "MIGS" / "MIGS" / "Task_I_annotation.json").exists(), reason="MIGS not available")


# ----------------------------------------------------------------------------- helpers
def check_invariants(recs: list[VideoRecord], dataset: str) -> None:
    """Shared VideoRecord invariants: ids, sorted segments, end >= start, JSON round trip."""
    assert recs
    assert len({r.sample_id for r in recs}) == len(recs)
    for r in recs:
        assert r.dataset == dataset
        assert r.sample_id == make_sample_id(dataset, r.video_id)
        assert SAMPLE_ID_RE.match(r.sample_id), r.sample_id
        assert r.source_kind in ("file", "zip_member", "concat_clips") and r.source_paths
        assert r.procedure and r.procedure_category in PROCEDURE_CATEGORIES
        assert r.license and r.citation
        assert "duration_bin" in r.strata
        starts = [s.start_s for s in r.segments]
        assert starts == sorted(starts)
        for s in r.segments:
            assert 0 <= s.start_s <= s.end_s, (r.sample_id, s)
            assert s.label and s.kind in ("phase", "operation", "step", "flag")
            if r.duration_s is not None:
                assert s.end_s <= r.duration_s + 1.0
        d = r.to_dict()
        json.dumps(d, ensure_ascii=False)
        assert VideoRecord.from_dict(d).to_dict() == d


def _write_csv(path: Path, header: list[str], rows: list[list], delimiter: str = ",") -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=delimiter)
        w.writerow(header)
        w.writerows(rows)


def _tmp_cfg(tmp_path: Path) -> Config:
    return Config(deep_merge(DEFAULTS, {"datasets_root": str(tmp_path)}), root=REPO)


@pytest.fixture(scope="module")
def c101():
    ad = get_adapter("cataract101", CFG)
    return ad, ad.records()


@pytest.fixture(scope="module")
def c1k():
    ad = get_adapter("cataract1k", CFG)
    return ad, ad.records()


@pytest.fixture(scope="module")
def migs():
    ad = get_adapter("migs", CFG)
    return ad, ad.records()


def test_registry_has_agent_a_adapters():
    assert {"cataract101", "cataract1k", "migs"} <= set(available_adapters())


# ----------------------------------------------------------------------------- cataract101
@needs_c101
class TestCataract101:
    def test_counts_and_invariants(self, c101):
        _, recs = c101
        assert len(recs) == 101
        check_invariants(recs, "cataract101")
        assert all(r.segments for r in recs)
        assert all(r.labels["n_phase_segments"] >= 10 for r in recs)
        assert all(r.source_kind == "file" and Path(r.source_paths[0]).name == f"{r.video_id}.mp4" for r in recs)

    def test_segment_rules_case_269(self, c101):
        _, recs = c101
        rec = {r.video_id: r for r in recs}["case_269"]
        assert rec.sample_id == "cataract101__case_269"
        assert rec.duration_s == pytest.approx(14734 / 25) and rec.fps == 25 and (rec.width, rec.height) == (720, 540)
        segs = rec.segments
        assert segs[0].label == LEAD_LABEL and segs[0].start_s == 0 and segs[0].end_s == pytest.approx(68 / 25)
        assert segs[1].label_id == 1 and segs[1].start_s == pytest.approx(68 / 25)
        assert segs[1].end_s == pytest.approx((1043 - 1) / 25)  # next start minus one frame
        assert [s.label_id for s in segs[1:]] == [1, 2, 3, 4, 5, 6, 7, 2, 8, 9, 10, 10]  # repeated phases kept
        assert segs[-1].end_s == pytest.approx(rec.duration_s)  # last segment runs to Frames
        assert rec.labels["repeated_phases"] == ["Tonifying and antibiotics", "Viscous agent injection"]
        assert rec.labels["n_phase_segments"] == 12 and rec.labels["n_distinct_phases"] == 10

    def test_leading_segment_and_gaps(self, c101):
        _, recs = c101
        assert sum(1 for r in recs if r.segments[0].label == LEAD_LABEL) == 97
        for r in recs:
            lead = r.segments[0].label == LEAD_LABEL
            assert lead == (r.labels["unannotated_lead_s"] > 0)
            for a, b in zip(r.segments, r.segments[1:]):
                gap = b.start_s - a.end_s
                assert -1e-6 <= gap <= 1 / 25 + 1e-6  # contiguous (lead) or one-frame gap

    def test_labels_and_strata(self, c101):
        _, recs = c101
        assert Counter(r.strata["experience"] for r in recs) == {"low": 45, "high": 56}
        assert Counter(r.strata["surgeon"] for r in recs) == {"1": 25, "2": 24, "3": 32, "4": 20}
        assert all(set(r.strata) == {"experience", "surgeon", "duration_bin"} for r in recs)
        assert all({"surgeon_id", "experience", "frames", "fps", "n_phase_segments"} <= set(r.labels) for r in recs)
        assert {r.strata["duration_bin"] for r in recs} <= {"<5min", "5-8min", "8-12min", ">12min"}
        assert all(r.labels["experience"] == ("low" if r.labels["experience_code"] == 1 else "high") for r in recs)
        assert all(r.procedure_category == "cataract" and "CC BY-NC" in r.license for r in recs)
        assert min(r.duration_s for r in recs) > 200 and max(r.duration_s for r in recs) < 1200

    def test_stats(self, c101):
        ad, recs = c101
        st = ad.stats(recs)
        assert st["n_records"] == 101 and st["n_with_duration"] == 101 and st["n_with_leading_unannotated"] == 97


# ----------------------------------------------------------------------------- cataract1k
@needs_c1k
class TestCataract1k:
    def test_counts(self, c1k):
        ad, recs = c1k
        assert len(recs) == 1000
        check_invariants(recs, "cataract1k")
        listed = {row["VideoID"] for row in csv.DictReader(open(DS / "cataract-1k/cataract-1k_annotations/videos.csv"))}
        on_disk = {e.name[5:-4] for e in os.scandir(DS / "cataract-1k/cat-1k") if e.name.endswith(".mp4")}
        assert len(listed) == 303
        annotated = [r for r in recs if r.labels["annotated"]]
        # DESIGN says 303, but only the annotated ids whose video exists on disk can become records (247 here).
        assert len(annotated) == len(listed & on_disk) >= 247
        assert ad.n_annotated_without_video == len(listed - on_disk)
        assert {r.video_id for r in recs} == {f"case_{v}" for v in on_disk}
        assert "cataract1k__case_2003" in {r.sample_id for r in recs}
        assert [r.video_id for r in recs] == sorted((r.video_id for r in recs), key=lambda v: int(v[5:]))

    def test_annotated_segments_at_1fps(self, c1k):
        _, recs = c1k
        by = {r.video_id: r for r in recs}
        rec = by["case_2003"]
        assert rec.duration_s == 414.0 and rec.labels["frames"] == 414 and rec.labels["annotation_fps"] == 1
        first, second, last = rec.segments[0], rec.segments[1], rec.segments[-1]
        assert (first.label, first.label_id, first.start_s, first.end_s) == ("Side Incision", 1, 1.0, 6.0)
        assert (second.label, second.start_s, second.end_s) == ("Idle", 6.0, 10.0)
        assert (last.label, last.end_s) == ("Wound Closure", 414.0)
        annotated = [r for r in recs if r.labels["annotated"]]
        for r in annotated:
            assert r.duration_s == r.labels["frames"]
            assert all(s.kind == "phase" and 0 <= s.label_id <= 13 for s in r.segments)
            for a, b in zip(r.segments, r.segments[1:]):
                assert b.start_s == pytest.approx(a.end_s)  # contiguous
            if r.segments:
                assert r.segments[-1].end_s == pytest.approx(r.duration_s)
            else:  # videos whose annotation holds only the end marker
                assert any("end-marker" in n for n in r.notes)
            assert r.labels["idle_share"] is None or 0 <= r.labels["idle_share"] <= 1
        assert sum(1 for r in annotated if r.labels["has_not_cataract"]) == 1

    def test_unannotated_and_size_bins(self, c1k):
        ad, recs = c1k
        unannotated = [r for r in recs if not r.labels["annotated"]]
        assert unannotated and all(r.duration_s is None and not r.segments for r in unannotated)
        assert all(r.strata["duration_bin"] == "unknown" for r in unannotated)
        assert all(r.labels["file_size_mb"] > 0 for r in recs)
        bins = Counter(r.strata["size_bin"] for r in recs)
        assert set(bins) == set(SIZE_BIN_LABELS) and all(200 <= n <= 300 for n in bins.values())
        assert all(set(r.strata) == {"annotated", "rare_flag", "duration_bin", "size_bin"} for r in recs)
        assert all(isinstance(r.strata["annotated"], bool) and isinstance(r.strata["rare_flag"], bool) for r in recs)
        required = {"annotated", "flags", "n_phase_segments", "has_not_cataract", "has_suture", "file_size_mb"}
        assert all(required <= set(r.labels) for r in recs)
        for r in recs:
            assert r.strata["rare_flag"] == bool(r.labels["flags"] or r.labels["has_suture"] or r.labels["has_not_cataract"])
        st = ad.stats(recs)
        assert st["n_records"] == 1000 and st["n_annotated"] == len(recs) - len(unannotated)
        assert len(st["size_bin_cuts_mb"]) == 3 and st["size_bin_cuts_mb"] == sorted(st["size_bin_cuts_mb"])


def test_cataract1k_flags_and_end_marker_synthetic(tmp_path):
    """Usage flags 14/15/16 -> kind=flag segments + labels.flags; end marker dropped; size bins over the dataset."""
    root = tmp_path / "cataract-1k"
    (root / "cat-1k").mkdir(parents=True)
    ann = root / "cataract-1k_annotations"
    ann.mkdir()
    for i, size in [(1, 1000), (2, 2000), (3, 3000), (4, 4000)]:
        (root / "cat-1k" / f"case_{i}.mp4").write_bytes(b"\0" * size)
    (root / "cat-1k" / "SYNAPSE_METADATA_MANIFEST.tsv").write_text("path\tname\n")
    _write_csv(ann / "videos.csv", ["VideoID", "Frames", "FPS"], [[1, 100, 1], [2, 50, 1], [9, 30, 1]])
    _write_csv(ann / "phases.csv", ["Phase", "Meaning"],
               [[0, "Idle"], [1, "Side Incision"], [3, "Capsulorhexis"], [10, "Not Cataract"], [13, "Suture"],
                [14, "Trypan Blue Injection Used"], [15, "Iris Hooks Used"], [16, "Malyugin Ring Used"]])
    _write_csv(ann / "annotations.csv", ["VideoID", "FrameNo", "Phase"],
               [[1, 0, 0], [1, 5, 1], [1, 10, 14], [1, 12, 3], [1, 40, 13], [1, 60, 16], [1, 61, 10], [1, 100, 0],
                [2, 0, 0], [2, 10, 3], [2, 50, 0],
                [9, 0, 0], [9, 30, 0]])
    ad = Cataract1kAdapter(_tmp_cfg(tmp_path))
    recs = ad.records()
    assert [r.video_id for r in recs] == ["case_1", "case_2", "case_3", "case_4"]
    by = {r.video_id: r for r in recs}
    r1 = by["case_1"]
    assert [(s.label_id, s.kind, s.start_s, s.end_s) for s in r1.segments] == [
        (0, "phase", 0.0, 5.0), (1, "phase", 5.0, 10.0), (14, "flag", 10.0, 12.0), (3, "phase", 12.0, 40.0),
        (13, "phase", 40.0, 60.0), (16, "flag", 60.0, 61.0), (10, "phase", 61.0, 100.0)]
    assert r1.labels["flags"] == ["Malyugin Ring Used", "Trypan Blue Injection Used"]
    assert r1.labels["has_suture"] and r1.labels["has_not_cataract"] and r1.strata["rare_flag"] is True
    assert r1.labels["n_phase_segments"] == 5 and r1.labels["n_flag_segments"] == 2 and len(r1.phase_segments) == 5
    assert r1.duration_s == 100.0 and r1.strata["duration_bin"] == "<5min" and not r1.notes
    r2 = by["case_2"]
    assert r2.strata["rare_flag"] is False and r2.labels["flags"] == [] and len(r2.segments) == 2
    assert all(not by[v].labels["annotated"] and by[v].duration_s is None for v in ("case_3", "case_4"))
    assert ad.n_annotated_without_video == 1  # id 9 has no video
    assert [by[f"case_{i}"].strata["size_bin"] for i in (1, 2, 3, 4)] == list(SIZE_BIN_LABELS)
    assert size_quartile_cuts([1, 2, 3]) is None and size_bin(5, None) == "unknown"


# ----------------------------------------------------------------------------- migs
@needs_migs
class TestMigs:
    def test_counts_and_sources(self, migs):
        _, recs = migs
        assert len(recs) == 185
        check_invariants(recs, "migs")
        assert Counter(r.source_kind for r in recs) == {"file": 10, "zip_member": 175}
        for r in recs:
            if r.source_kind == "file":
                assert Path(r.source_paths[0]).is_file()
            else:
                zp, member = r.source_paths[0].split("!", 1)
                assert zp.endswith(".zip") and Path(zp).is_absolute() and Path(zp).is_file() and member.endswith(".mp4")
        for r in [r for r in recs if r.source_kind == "zip_member"][:3]:  # spot-check zip tables of contents
            zp, member = r.source_paths[0].split("!", 1)
            assert member in zipfile.ZipFile(zp).namelist()
        by = {r.video_id: r for r in recs}
        assert by["284_1"].sample_id == "migs__284_1" and by["284_1"].source_paths[0].endswith(".zip!284 1.mp4")
        assert by["16"].source_kind == "file" and by["16"].source_paths[0].endswith("/16.mp4")
        assert any("key is used as the id" in n for n in by["284_1"].notes) and by["284_1"].labels["json_video_id"] == 284

    def test_resolve_files(self):
        files = resolve_migs_files(DS / "MIGS" / "MIGS")
        data = json.load(open(DS / "MIGS" / "MIGS" / "Task_I_annotation.json", encoding="utf-8"))
        assert set(data) <= set(files)
        assert len(files) == 206 and len(set(files) - set(data)) == 21
        assert files["284_1"]["source_kind"] == "zip_member" and files["284_1"]["member"] == "284 1.mp4"
        assert files["16"]["source_kind"] == "file" and files["16"]["zip"] is None
        assert all(v["size_bytes"] > 0 and len(v["source_paths"]) == 1 for v in files.values())

    def test_labels_segments_strata(self, migs):
        ad, recs = migs
        labels = {s.label for r in recs for s in r.segments}
        assert "Would closure" not in labels and "Wound closure" in labels and len(labels) == 8
        assert all(s.kind == "step" and s.label_id in range(8) for r in recs for s in r.segments)
        overlapping = [r for r in recs if r.labels["has_overlapping_segments"]]
        assert overlapping
        r = overlapping[0]
        assert any(b.start_s < a.end_s for a, b in zip(r.segments, r.segments[1:]))  # overlaps are kept
        assert all(r.duration_s and r.fps and r.segments for r in recs)
        assert all(set(r.strata) == {"operation_type", "knife", "duration_bin"} for r in recs)
        assert all(r.procedure == f"Glaucoma MIGS: {r.labels['operation_type']}" and r.procedure_category == "glaucoma"
                   for r in recs)
        assert all(isinstance(r.labels["operation_type_glossary"], dict) and r.labels["operation_type_glossary"] for r in recs)
        ops = Counter(r.strata["operation_type"] for r in recs)
        assert ops["PEI_GT120"] == 56 and ops["GT120"] == 46 and len(ops) == 13
        assert {r.strata["knife"] for r in recs} <= {"TMH", "KDB", "cystotome_needle", "mixed", "unknown"}
        assert Counter(r.strata["knife"] for r in recs)["TMH"] == 158
        for r in recs:
            assert r.labels["gt_incision_num"] is None or isinstance(r.labels["gt_incision_num"], int)
            if r.labels["gt_incision_note"]:
                assert r.labels["gt_incision_note_en"]  # every Chinese free-text note has a translation
        st = ad.stats(recs)
        assert st["n_records"] == 185 and len(st["unannotated_files"]) == 21 and st["source_kinds"]["file"] == 10


def test_migs_helpers():
    assert normalise_label("Would closure") == "Wound closure" and normalise_label(" Gonioscopy ") == "Gonioscopy"
    assert normalise_video_key("284 1.mp4") == "284_1" and normalise_video_key("/a/b/16.mp4") == "16"
    assert normalise_video_key("313_Full_Version.mp4") == "313_Full_Version"
    op = parse_operation_type("PEI_GSL_GT120")
    assert op["components"] == ["PEI", "GSL", "GT120"] and op["goniotomy_degrees"] == 120
    assert op["combined_with_phaco"] and op["has_goniosynechialysis"] and set(op["glossary"]) == {"PEI", "GSL", "GT"}
    op2 = parse_operation_type("PEI+GT")
    assert op2["components"] == ["PEI", "GT"] and op2["goniotomy_degrees"] is None
    op3 = parse_operation_type("GATT360")
    assert op3["has_gatt"] and op3["goniotomy_degrees"] is None and "GATT" in op3["glossary"]
    assert parse_operation_type(None)["code"] == "unknown"
    assert normalise_knife("KBD") == "KDB" and normalise_knife("TMH") == "TMH" and normalise_knife(None) == "unknown"
    assert normalise_knife("KDB&TMH") == "mixed" and normalise_knife("破囊针头") == "cystotome_needle"


def test_migs_synthetic(tmp_path):
    """Loose file vs zip member resolution, label fix, overlaps, missing-file skip, knife/incision normalisation."""
    root = tmp_path / "MIGS" / "MIGS"
    root.mkdir(parents=True)
    (root / "16.mp4").write_bytes(b"x")
    with zipfile.ZipFile(root / "MIGS_video_dataset_2.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("284 1.mp4", b"yy")
        zf.writestr("105.mp4", b"zzz")
    seg = lambda label, lid, s, e: {"label": label, "label_id": lid, "segment": [s, e], "segment(frames)": [s * 25, e * 25]}  # noqa: E731
    data = {
        "16": {"annotations": [seg("Gonioscopy", 2, 10.0, 50.0), seg("Would closure", 5, 60.0, 70.0), seg("Goniotomy", 3, 20.0, 45.0)],
               "video_id": 16, "clear": True, "operation type": "GT120", "knife": "KBD", "GT incision num": "看不到GT",
               "subset": "training", "fps": 25.0, "duration": 80.0, "frame_count": 2000},
        "284_1": {"annotations": [seg("OVDs injection", 1, 5.0, 8.0)], "video_id": 284, "clear": False,
                  "operation type": "PEI+GT", "subset": "testing", "fps": 30.0, "duration": 100.0, "frame_count": 3000},
        "999": {"annotations": [], "video_id": 999, "operation type": "GT240", "fps": 25.0, "duration": 10.0, "frame_count": 250},
    }
    (root / "Task_I_annotation.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    ad = MigsAdapter(_tmp_cfg(tmp_path))
    recs = ad.records()
    assert [r.video_id for r in recs] == ["16", "284_1"]  # 999 has no file -> skipped with a warning
    r16, r284 = recs
    assert r16.source_kind == "file" and r16.source_paths == [str((root / "16.mp4").resolve())]
    assert [s.label for s in r16.segments] == ["Gonioscopy", "Goniotomy", "Wound closure"]  # sorted, typo fixed
    assert r16.segments[2].extra["original_label"] == "Would closure" and r16.labels["has_overlapping_segments"]
    assert r16.labels["knife_norm"] == "KDB" and r16.strata["knife"] == "KDB"
    assert r16.labels["gt_incision_num"] is None and r16.labels["gt_incision_note_en"] == "goniotomy not visible in the video"
    assert r16.strata["duration_bin"] == "<5min" and r16.labels["goniotomy_degrees"] == 120
    assert r284.source_kind == "zip_member" and r284.source_paths[0].endswith("MIGS_video_dataset_2.zip!284 1.mp4")
    assert r284.labels["knife"] is None and r284.labels["knife_norm"] == "unknown"
    assert r284.procedure == "Glaucoma MIGS: PEI+GT" and r284.labels["goniotomy_degrees"] is None
    assert any("key is used as the id" in n for n in r284.notes)
    assert ad.unannotated_files == ["105"]
    check_invariants(recs, "migs")
