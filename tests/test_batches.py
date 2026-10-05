"""Offline tests for bench.batches and the batch modes of export-ui / generate / prepare (DESIGN.md 13).

Synthetic data only (PIL contact sheets, tiny ffmpeg test clips); no datasets, no network.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench import batches, export_ui, prepare, sampler  # noqa: E402
from bench.config import load_config  # noqa: E402
from bench.schema import FrameInfo, PreparedSample, ProbeInfo, QAItem, QASet, SampleRecord, Segment  # noqa: E402
from bench.util import read_json, write_json, write_jsonl  # noqa: E402


# ------------------------------------------------------------------------------------ fixtures
@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("OPHBENCH_DATA_DIR", str(data))
    monkeypatch.setenv("OPHBENCH_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("OPHBENCH_UI_DIR", str(tmp_path / "ui"))
    monkeypatch.setenv("OPHBENCH_PACKAGES_DIR", str(tmp_path / "ui_packages"))
    cfg = load_config()
    cfg.ensure_dirs()
    return cfg


def rec(dataset: str, i: int, **kw) -> SampleRecord:
    base = dict(source_kind="file", source_paths=[f"/x/{dataset}/{i}.mp4"], duration_s=100.0 + i, procedure="P",
                procedure_category="cataract", segments=[Segment(label="A", start_s=0, end_s=50), Segment(label="B", start_s=50, end_s=100)],
                labels={"experience": "high"}, stratum_key="s", selection_reason="r", sample_index=-1)
    base.update(kw)
    return SampleRecord(sample_id=f"{dataset}__v{i}", dataset=dataset, video_id=f"v{i}", **base)


def write_prepared(cfg, r: SampleRecord, *, with_preview: bool) -> Path:
    d = Path(cfg.paths.prepared) / r.sample_id
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (10, 20, 30)).save(d / "contact_sheet.jpg", "JPEG")
    (d / "source.mp4").write_bytes(b"S" * 300)
    if with_preview:
        (d / "preview.mp4").write_bytes(b"P" * 120)
    ps = PreparedSample(record=r, probe=ProbeInfo(duration_s=float(r.duration_s or 100.0), fps=25.0, width=640, height=360),
                        frames=[FrameInfo(idx=0, t_s=1.0, file="frames/f_000_0001.0s.jpg", label_at_t="A")],
                        preview="preview.mp4" if with_preview else None, timeline_note="note", prepared_at="t")
    write_json(d / "sample.json", ps.to_dict())
    return d


def write_qa(cfg, sid: str) -> None:
    q = lambda qid, cat: QAItem(qid=qid, category=cat, question=f"Q {qid}?", answer=f"A {qid}", answer_rationale="r",  # noqa: E731
                                evidence_timestamps=[{"start_s": 1, "end_s": 2, "observation": "o"}], confidence=0.9)
    qs = QASet(sample_id=sid, video_summary="s", questions=[q("q1", "temporal_grounding"), q("q2", "workflow_deviation"),
                                                            q("q3", "quantitative_estimation")],
               provenance={"generated_at": "2026-10-05T12:00:00"})
    write_json(Path(cfg.paths.qa) / f"{sid}.json", qs.to_dict())


def export_args(**kw) -> argparse.Namespace:
    base = dict(command="export-ui", media_mode=None, annotators=None, overlap=None, include_invalid=False, batch=None,
                batch_size=None, no_package=False, package_kind=None, packages_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


# ------------------------------------------------------------------------------------ order / batches
def test_order_samples_interleaves_proportionally():
    records = [rec("a", i) for i in range(60)] + [rec("b", i) for i in range(30)] + [rec("c", i) for i in range(10)]
    ordered = batches.order_samples(records)
    assert len(ordered) == 100 and len({r.sample_id for r in ordered}) == 100
    first_half = Counter(r.dataset for r in ordered[:50])
    assert first_half["a"] == 30 and first_half["b"] == 15 and first_half["c"] == 5       # exact proportions
    # within a dataset the input order is kept
    assert [r.video_id for r in ordered if r.dataset == "c"] == [f"v{i}" for i in range(10)]
    # deterministic and independent of sample_index values / input order across datasets
    shuffled = list(reversed(records))
    assert [r.sample_id for r in batches.order_samples(shuffled) if r.dataset == "a"] == [f"a__v{i}" for i in reversed(range(60))]
    assert batches.order_samples(records) == ordered
    # dict rows work too
    rows = [{"sample_id": r.sample_id, "dataset": r.dataset} for r in records]
    assert [d["sample_id"] for d in batches.order_samples(rows)] == [r.sample_id for r in ordered]


def test_select_batch_and_assign():
    records = [rec("a", i) for i in range(7)] + [rec("b", i) for i in range(5)]
    ordered = batches.order_samples(records)
    assert batches.n_batches(12, 5) == 3 and batches.n_batches(0, 5) == 0
    b1, b2, b3 = (batches.select_batch(ordered, k, 5) for k in (1, 2, 3))
    assert [len(b1), len(b2), len(b3)] == [5, 5, 2] and batches.select_batch(ordered, 4, 5) == []
    assert b1 + b2 + b3 == ordered
    with pytest.raises(ValueError):
        batches.select_batch(ordered, 0, 5)
    by = batches.assign_batches(records, 5)
    assert Counter(by.values()) == {1: 5, 2: 5, 3: 2}
    assert all(by[r.sample_id] == 1 for r in b1) and all(by[r.sample_id] == 3 for r in b3)
    assert batches.batch_of(0, 100) == 1 and batches.batch_of(99, 100) == 1 and batches.batch_of(100, 100) == 2
    desc = batches.describe(records, 5)
    assert [d["n"] for d in desc] == [5, 5, 2] and desc[0]["datasets"] == {"a": 3, "b": 2}


def test_batch_size_and_package_kind(env):
    cfg = env
    assert batches.batch_size(cfg) == int(cfg.get("batches.size", 100) or 100)
    assert batches.batch_size(cfg, 7) == 7 and batches.batch_size(None) == 100
    assert batches.package_kind(cfg) in batches.PACKAGE_KINDS and batches.package_kind(cfg, "source") == "source"
    with pytest.raises(ValueError):
        batches.package_kind(cfg, "bogus")
    assert batches.batch_dir_name(3) == "batch_003"


def test_sampler_manifest_is_in_benchmark_order():
    """build_manifest writes the canonical order with sample_index == rank."""
    picks = {ds: [sampler.Pick(rec(ds, i), "s", "why") for i in range(n)] for ds, n in (("a", 4), ("b", 2))}
    outcomes = {ds: sampler.DatasetOutcome(sampler.DatasetPlan(ds, len(p), [], [], [], lambda r: "s"), p) for ds, p in picks.items()}
    result = sampler.SamplingResult(6, 1, {"a": 4, "b": 2}, {"a": 4, "b": 2}, outcomes, [], [])
    manifest = sampler.build_manifest(result)
    assert [m.sample_index for m in manifest] == list(range(6))
    assert [m.sample_id for m in manifest] == [m.sample_id for m in batches.order_samples(manifest)]
    assert [m.dataset for m in manifest] == ["a", "b", "a", "a", "b", "a"]


# ------------------------------------------------------------------------------------ export-ui --batch
def test_export_batches_are_additive_and_packaged(env, tmp_path):
    cfg = env
    records = [rec("a", i) for i in range(4)] + [rec("b", i) for i in range(2)]
    ordered = batches.order_samples(records)
    write_jsonl(Path(cfg.paths.sample) / "sample_manifest.jsonl", [r.to_dict() for r in records])
    for r in records:
        write_prepared(cfg, r, with_preview=(r.dataset == "a"))
        write_qa(cfg, r.sample_id)
    ui = export_ui.ui_paths(cfg)

    assert export_ui.main(export_args(batch=1, batch_size=3, annotators="x,y", overlap=0.34), cfg) == 0
    index = read_json(ui.data / "index.json")
    assert [e["sample_id"] for e in index] == [r.sample_id for r in ordered[:3]]
    assert all(e["batch"] == 1 for e in index)
    assert all(e["preview"] is None for e in index)                      # sheets mode: videos are offline
    assert all((ui.media / e["sample_id"] / "contact_sheet.jpg").is_file() for e in index)
    assert not any((ui.media / e["sample_id"] / "preview.mp4").exists() for e in index)
    meta = read_json(ui.data / "meta.json")
    assert meta["n_samples"] == 3 and meta["batch_size"] == 3 and meta["n_batches_total"] == 2
    assert meta["offline_media"] is True and meta["batches"]["1"]["n"] == 3
    doc = read_json(ui.samples / f"{index[0]['sample_id']}.json")
    assert doc["batch"] == 1 and doc["video"]["local_file"] == f"{index[0]['sample_id']}.mp4" and doc["video"]["preview"] is None
    pkg = Path(meta["batches"]["1"]["package"])
    assert pkg == tmp_path / "ui_packages" / "batch_001"
    files = sorted(p.name for p in pkg.iterdir())
    assert "MANIFEST.csv" in files and "README.txt" in files
    assert sorted(f for f in files if f.endswith(".mp4")) == sorted(f"{e['sample_id']}.mp4" for e in index)
    # dataset 'a' had previews (120 bytes); 'b' falls back to the 300-byte source
    sizes = {p.name: p.stat().st_size for p in pkg.glob("*.mp4")}
    for e in index:
        assert sizes[f"{e['sample_id']}.mp4"] == (120 if e["dataset"] == "a" else 300)
    manifest_csv = (pkg / "MANIFEST.csv").read_text().splitlines()
    assert manifest_csv[0] == ",".join(export_ui.PACKAGE_MANIFEST_FIELDS) and len(manifest_csv) == 4
    assert "Open batch folder" in (pkg / "README.txt").read_text()
    assert all((pkg / f"{e['sample_id']}.qa.json").is_file() for e in index)   # GPT questions travel with the videos
    asg1 = read_json(ui.data / "assignments.json")
    assert asg1["annotators"] == ["x", "y"] and len(asg1["overlap"]) == 2            # ceil(0.34*3) = 2

    # batch 2 is added without touching batch 1
    assert export_ui.main(export_args(batch=2, batch_size=3, annotators="x,y", overlap=0.34), cfg) == 0
    index2 = read_json(ui.data / "index.json")
    assert [e["sample_id"] for e in index2] == [r.sample_id for r in ordered]
    assert [e["batch"] for e in index2] == [1, 1, 1, 2, 2, 2]
    meta2 = read_json(ui.data / "meta.json")
    assert meta2["n_samples"] == 6 and sorted(meta2["batches"]) == ["1", "2"] and meta2["batches"]["1"]["n"] == 3
    asg2 = read_json(ui.data / "assignments.json")
    assert set(asg2["by_annotator"]["x"]) | set(asg2["by_annotator"]["y"]) == {r.sample_id for r in records}
    assert asg2["overlap"][:2] == asg1["overlap"]                                     # earlier batch kept
    assert (tmp_path / "ui_packages" / "batch_002" / "MANIFEST.csv").exists()
    assert all((ui.samples / f"{r.sample_id}.json").exists() for r in records)
    # re-exporting batch 1 is idempotent (same entries, package files not re-copied)
    before = {p.name: p.stat().st_mtime_ns for p in pkg.glob("*.mp4")}
    assert export_ui.main(export_args(batch=1, batch_size=3, annotators="x,y", overlap=0.34), cfg) == 0
    assert read_json(ui.data / "index.json") == index2
    assert {p.name: p.stat().st_mtime_ns for p in pkg.glob("*.mp4")} == before


def test_export_without_batch_keeps_full_rebuild(env, tmp_path):
    cfg = env
    records = [rec("a", 0), rec("b", 0)]
    write_jsonl(Path(cfg.paths.sample) / "sample_manifest.jsonl", [r.to_dict() for r in records])
    for r in records:
        write_prepared(cfg, r, with_preview=True)
        write_qa(cfg, r.sample_id)
    ui = export_ui.ui_paths(cfg)
    assert export_ui.main(export_args(no_package=True), cfg) == 0
    index = read_json(ui.data / "index.json")
    assert len(index) == 2 and all(e["preview"] == f"media/{e['sample_id']}/preview.mp4" for e in index)   # symlink default
    meta = read_json(ui.data / "meta.json")
    assert meta["offline_media"] is False and meta["batches"] == {} and all(e["batch"] == 1 for e in index)
    assert not (tmp_path / "ui_packages").exists()


# ------------------------------------------------------------------------------------ generate --batch (selection)
def test_generate_batch_selection(env, monkeypatch):
    """--batch narrows the manifest before the usual prepared/existing filters (dry-run, no API)."""
    from bench import generate

    cfg = env
    records = [rec("a", i) for i in range(4)] + [rec("b", i) for i in range(2)]
    write_jsonl(Path(cfg.paths.sample) / "sample_manifest.jsonl", [r.to_dict() for r in records])
    ordered = batches.order_samples(records)
    seen: list[str] = []

    def fake_dry_run(item, cfg_, opts, log):  # noqa: ANN001
        seen.append(item.record.sample_id)
        return generate.GenerationLogEntry(sample_id=item.record.sample_id, model="m", route="responses",
                                           reasoning_effort="low", n_frames=1, status="dry_run")

    monkeypatch.setattr(generate, "dry_run_one", fake_dry_run)
    for r in records:
        write_prepared(cfg, r, with_preview=False)
    args = argparse.Namespace(concurrency=1, limit=None, ids=None, datasets=None, dry_run=True, force=False, route=None,
                              effort=None, max_tokens=None, batch=2, batch_size=4)
    assert generate.main(args, cfg) == 0
    assert sorted(seen) == sorted(r.sample_id for r in ordered[4:])
    args.batch = 9
    assert generate.main(args, cfg) == 0                                             # empty batch: warning, rc 0


# ------------------------------------------------------------------------------------ prepare: add missing previews
def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def test_prepare_adds_missing_preview_without_force(env, tmp_path):
    cfg = env
    clip = tmp_path / "clip.mp4"
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "2",
                    "-pix_fmt", "yuv420p", str(clip)], check=True, timeout=120)
    r = rec("a", 0, source_paths=[str(clip)], duration_s=None, segments=[])
    write_jsonl(Path(cfg.paths.sample) / "sample_manifest.jsonl", [r.to_dict()])
    base = dict(workers=1, ids=None, datasets=None, limit=None, force=False, manifest=None, batch=None, batch_size=None)
    assert prepare.main(argparse.Namespace(no_preview=True, **base), cfg) == 0
    sdir = Path(cfg.paths.prepared) / "batch_001" / r.sample_id   # per-batch layout
    assert (sdir / "sample.json").exists() and not (sdir / "preview.mp4").exists()
    frames_before = sorted(p.name for p in (sdir / "frames").iterdir())
    assert prepare.main(argparse.Namespace(no_preview=False, **base), cfg) == 0
    assert (sdir / "preview.mp4").stat().st_size > 0
    assert read_json(sdir / "sample.json")["preview"] == "preview.mp4"
    assert sorted(p.name for p in (sdir / "frames").iterdir()) == frames_before          # frames untouched
    rows = [json.loads(l) for l in (Path(cfg.paths.prepared) / "_status.jsonl").read_text().splitlines()]
    assert rows[-1]["steps_done"] == ["preview"] and rows[-1]["ok"] and rows[-1].get("mode") == "preview_only"
    # third run: nothing to do
    assert prepare.main(argparse.Namespace(no_preview=False, **base), cfg) == 0
    assert len([json.loads(l) for l in (Path(cfg.paths.prepared) / "_status.jsonl").read_text().splitlines()]) == 2
