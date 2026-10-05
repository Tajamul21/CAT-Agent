"""Offline tests for bench.media (frame planning, probing, overlays) and bench.prepare helpers.

No network; the only dataset access is one 6 s Ophora clip (skipped when not mounted).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from bench import media
from bench.config import load_config
from bench.prepare import (build_timeline_note, canonical_order, label_at_t, order_samples_fallback,
                           realign_concat_segments, select_batch, select_records)
from bench.schema import ProbeInfo, SampleRecord, Segment

OPHORA_CLIP = Path("/mnt/store/tashraf4/datasets/Ophora-160K/clips/X3jTUYMflCk_26.mp4")
FRAMES_CFG = {"frames": {"seconds_per_frame": 10, "min_frames": 12, "max_frames": 48, "resize_long_side": 768,
                         "jpeg_quality": 85, "boundary_frames": True}}

BANNER = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/x/X3jTUYMflCk_26.mp4':
  Metadata:
    major_brand     : isom
    encoder         : Lavf58.29.100
  Duration: 00:00:06.01, start: 0.000000, bitrate: 3015 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080, 3011 kb/s, 29.97 fps, 29.97 tbr, 11988 tbn (default)
      Metadata:
        handler_name    : VideoHandler
At least one output file must be specified
"""


def _times(plan):
    return [t for t, _ in plan]


def _reasons(plan):
    return [r for _, r in plan]


def _assert_well_formed(plan, duration):
    ts = _times(plan)
    assert ts == sorted(ts), "times must be sorted"
    assert len(set(ts)) == len(ts), "times must be unique"
    assert all(0.0 <= t < duration for t in ts), "times must lie inside the video"
    assert set(_reasons(plan)) <= {"grid", "boundary", "midpoint"}


# ------------------------------------------------------------------------------------ plan_frame_times
class TestPlanFrameTimes:
    def test_short_clip_gets_min_frames(self):
        plan = media.plan_frame_times(6.0, [], FRAMES_CFG)
        assert len(plan) == 12
        assert set(_reasons(plan)) == {"grid"}
        _assert_well_formed(plan, 6.0)

    def test_two_minutes_rounds_to_min(self):
        assert len(media.plan_frame_times(120.0, None, FRAMES_CFG)) == 12

    def test_three_minutes(self):
        assert len(media.plan_frame_times(180.0, None, FRAMES_CFG)) == 18

    def test_ten_minutes_capped(self):
        assert len(media.plan_frame_times(600.0, None, FRAMES_CFG)) == 48

    def test_two_hours_capped(self):
        plan = media.plan_frame_times(7200.0, None, FRAMES_CFG)
        assert len(plan) == 48
        _assert_well_formed(plan, 7200.0)

    def test_grid_positions(self):
        cfg = {"frames": {"seconds_per_frame": 10, "min_frames": 4, "max_frames": 10}}
        plan = media.plan_frame_times(100.0, None, cfg)
        assert _times(plan) == pytest.approx([5 + 10 * i for i in range(10)])
        assert set(_reasons(plan)) == {"grid"}

    def test_accepts_frames_dict_directly(self):
        plan_a = media.plan_frame_times(300.0, None, FRAMES_CFG)
        plan_b = media.plan_frame_times(300.0, None, FRAMES_CFG["frames"])
        assert plan_a == plan_b

    def test_zero_negative_or_none_duration(self):
        assert media.plan_frame_times(0.0, None, FRAMES_CFG) == []
        assert media.plan_frame_times(-5.0, None, FRAMES_CFG) == []
        assert media.plan_frame_times(None, None, FRAMES_CFG) == []

    def test_segments_add_boundary_and_midpoint(self):
        segs = [Segment("A", 1, 0.0, 100.0), Segment("B", 2, 100.0, 300.0), Segment("C", 3, 300.0, 600.0)]
        plan = media.plan_frame_times(600.0, segs, FRAMES_CFG)
        _assert_well_formed(plan, 600.0)
        assert len(plan) == 48  # 48 grid + 6 anchors, trimmed back to max_frames
        by_reason = {r: [t for t, rr in plan if rr == r] for r in ("boundary", "midpoint", "grid")}
        assert by_reason["boundary"] == pytest.approx([1.0, 101.0, 301.0])
        assert by_reason["midpoint"] == pytest.approx([50.0, 200.0, 450.0])
        assert len(by_reason["grid"]) == 42

    def test_dedupe_prefers_boundary_over_grid(self):
        # grid points for 600 s are 6.25 + 12.5*i, so a segment starting at 92.75 puts its boundary on 93.75
        segs = [Segment("Phaco", 5, 92.75, 600.0)]
        plan = media.plan_frame_times(600.0, segs, FRAMES_CFG)
        near = [(t, r) for t, r in plan if abs(t - 93.75) < 1.5]
        assert near == [(93.75, "boundary")]
        assert len(plan) == 48

    def test_dense_segments_capped_and_spread(self):
        segs = [Segment(f"p{i}", i, 5.0 * i, 5.0 * (i + 1)) for i in range(60)]
        plan = media.plan_frame_times(300.0, segs, FRAMES_CFG)
        _assert_well_formed(plan, 300.0)
        assert len(plan) == 48
        ts = _times(plan)
        assert min(b - a for a, b in zip(ts, ts[1:])) >= 1.5 - 1e-6

    def test_boundary_frames_disabled(self):
        cfg = {"frames": dict(FRAMES_CFG["frames"], boundary_frames=False)}
        segs = [Segment("A", 1, 0.0, 300.0), Segment("B", 2, 300.0, 600.0)]
        plan = media.plan_frame_times(600.0, segs, cfg)
        assert set(_reasons(plan)) == {"grid"}
        assert len(plan) == 48

    def test_short_segment_boundary_stays_inside(self):
        segs = [Segment("blip", 9, 10.0, 10.6)]
        plan = media.plan_frame_times(200.0, segs, FRAMES_CFG)
        bts = [t for t, r in plan if r == "boundary"]
        assert bts == pytest.approx([10.3])

    def test_dict_and_tuple_segments(self):
        seg_objs = [Segment("A", 1, 0.0, 100.0), Segment("B", 2, 100.0, 250.0)]
        seg_dicts = [{"start_s": 0.0, "end_s": 100.0}, {"start_s": 100.0, "end_s": 250.0}]
        seg_tuples = [(0.0, 100.0), (100.0, 250.0)]
        a = media.plan_frame_times(250.0, seg_objs, FRAMES_CFG)
        assert a == media.plan_frame_times(250.0, seg_dicts, FRAMES_CFG)
        assert a == media.plan_frame_times(250.0, seg_tuples, FRAMES_CFG)

    def test_deterministic(self):
        segs = [Segment(f"p{i}", i, 37.0 * i, 37.0 * (i + 1)) for i in range(15)]
        assert media.plan_frame_times(589.36, segs, FRAMES_CFG) == media.plan_frame_times(589.36, segs, FRAMES_CFG)

    @pytest.mark.parametrize("duration", [3.0, 7.5, 45.0, 130.0, 589.36, 1500.0, 7200.0])
    def test_count_within_bounds(self, duration):
        for segs in ([], [Segment(f"p{i}", i, duration * i / 12, duration * (i + 1) / 12) for i in range(12)]):
            plan = media.plan_frame_times(duration, segs, FRAMES_CFG)
            _assert_well_formed(plan, duration)
            assert 12 <= len(plan) <= 48

    def test_segments_beyond_duration_are_clamped(self):
        segs = [Segment("late", 1, 500.0, 900.0)]
        plan = media.plan_frame_times(600.0, segs, FRAMES_CFG)
        _assert_well_formed(plan, 600.0)


# ------------------------------------------------------------------------------------ probing
class TestProbe:
    def test_parse_ffmpeg_banner(self):
        info = media.parse_ffmpeg_banner(BANNER)
        assert info.duration_s == pytest.approx(6.01)
        assert (info.width, info.height) == (1920, 1080)
        assert info.fps == pytest.approx(29.97)
        assert info.codec == "h264"

    def test_parse_banner_hours_and_tbr(self):
        text = "  Duration: 01:02:03.50, start: 0.000000\n  Stream #0:0: Video: hevc (Main), yuv420p, 720x480, 25 tbr\n"
        info = media.parse_ffmpeg_banner(text)
        assert info.duration_s == pytest.approx(3723.5)
        assert (info.width, info.height, info.fps, info.codec) == (720, 480, 25.0, "hevc")

    def test_parse_banner_empty(self):
        assert media.parse_ffmpeg_banner("").duration_s == 0.0

    def test_fraction_parsing(self):
        assert media._parse_fraction("2997/100") == pytest.approx(29.97)
        assert media._parse_fraction("25/1") == 25.0
        assert media._parse_fraction("0/0") is None
        assert media._parse_fraction("N/A") is None

    def test_missing_file_raises(self):
        with pytest.raises(media.MediaError):
            media.ffprobe("/nonexistent/video.mp4", load_config())

    @pytest.mark.skipif(not OPHORA_CLIP.exists(), reason="Ophora clip not available")
    def test_ffprobe_ophora_clip(self):
        info = media.ffprobe(OPHORA_CLIP, load_config())
        assert isinstance(info, ProbeInfo)
        assert info.duration_s == pytest.approx(6.0, abs=0.2)
        assert (info.width, info.height) == (1920, 1080)
        assert info.fps == pytest.approx(29.97, abs=0.1)
        assert info.codec == "h264"
        assert info.size_bytes == OPHORA_CLIP.stat().st_size

    @pytest.mark.skipif(not OPHORA_CLIP.exists(), reason="Ophora clip not available")
    def test_ffprobe_fallback_to_ffmpeg_banner(self):
        cfg = load_config()
        cfg.raw["ffprobe"] = "/nonexistent/ffprobe"  # Config.ffprobe() -> None -> banner fallback
        assert cfg.ffprobe() is None
        info = media.ffprobe(OPHORA_CLIP, cfg)
        assert info.duration_s == pytest.approx(6.0, abs=0.2)
        assert (info.width, info.height) == (1920, 1080)
        assert info.codec == "h264"


# ------------------------------------------------------------------------------------ overlays
def _gray_jpeg(path: Path, size=(768, 432), value=128) -> Path:
    Image.new("RGB", size, (value, value, value)).save(path, "JPEG", quality=90)
    return path


class TestOverlayAndContactSheet:
    def test_overlay_text_draws_black_box_with_white_text(self, tmp_path):
        src = _gray_jpeg(tmp_path / "f.jpg")
        out = media.overlay_text(src, ["t=00:12.5  (frame 3/48)", "Phacoemulsification"], out_path=tmp_path / "o.jpg")
        with Image.open(out) as im:
            assert im.size == (768, 432)
            w, h = im.size
            bl = im.crop((0, h // 2, w // 2, h)).convert("L")
            lo, hi = bl.getextrema()
            assert lo < 30, "black box expected in the bottom-left"
            assert hi > 200, "white text expected in the bottom-left"
            tr = im.crop((w // 2, 0, w, h // 2)).convert("L")
            assert tr.getextrema() == (128, 128), "top-right must be untouched"

    def test_overlay_in_place_and_empty_lines(self, tmp_path):
        src = _gray_jpeg(tmp_path / "f.jpg")
        assert media.overlay_text(src, ["", None]) == src  # type: ignore[list-item]
        with Image.open(src) as im:
            assert im.convert("L").getextrema() == (128, 128)

    def test_build_contact_sheet(self, tmp_path):
        frames = [_gray_jpeg(tmp_path / f"r{i}.jpg", size=(640, 360)) for i in range(7)]
        labels = [[f"t=00:0{i}.0  (frame {i + 1}/7)", "Idle"] for i in range(7)]
        out = media.build_contact_sheet(frames, tmp_path / "sheet.jpg", cols=6, tile_width=160, labels=labels)
        with Image.open(out) as im:
            assert im.size == (6 * 160, 2 * 90)
            tile0 = im.crop((0, 45, 80, 90)).convert("L")
            assert tile0.getextrema()[0] < 30  # overlay box drawn at tile scale

    def test_contact_sheet_fewer_frames_than_cols(self, tmp_path):
        frames = [_gray_jpeg(tmp_path / f"r{i}.jpg", size=(400, 300)) for i in range(3)]
        out = media.build_contact_sheet(frames, tmp_path / "sheet.jpg", cols=6, tile_width=100)
        with Image.open(out) as im:
            assert im.size == (300, 75)

    def test_contact_sheet_empty_raises(self, tmp_path):
        with pytest.raises(media.MediaError):
            media.build_contact_sheet([], tmp_path / "sheet.jpg")


# ------------------------------------------------------------------------------------ prepare helpers
def _rec(sample_id: str, dataset: str, **kw) -> SampleRecord:
    return SampleRecord(sample_id=sample_id, dataset=dataset, video_id=sample_id.split("__", 1)[1], **kw)


class TestPrepareHelpers:
    def test_label_at_t(self):
        segs = [Segment("Idle", 0, 0.0, 2.7), Segment("Incision", 1, 2.7, 41.7),
                Segment("Gonioscopy", 3, 10.0, 30.0, kind="step"), Segment("Goniotomy", 4, 15.0, 20.0, kind="step")]
        assert label_at_t(segs, 1.0) == "Idle"
        assert label_at_t(segs, 5.0) == "Incision"
        assert label_at_t(segs, 17.0) == "Goniotomy | Gonioscopy"
        assert label_at_t(segs, 41.7) == "Incision"  # inclusive end fallback
        assert label_at_t(segs, 100.0) is None
        assert label_at_t([], 1.0) is None

    def test_select_records(self):
        recs = [_rec("a__1", "a"), _rec("b__2", "b"), _rec("a__3", "a"), _rec("c__4", "c")]
        assert [r.sample_id for r in select_records(recs, datasets=["a"])] == ["a__1", "a__3"]
        assert [r.sample_id for r in select_records(recs, ids=["c__4", "a__1"])] == ["a__1", "c__4"]
        assert [r.sample_id for r in select_records(recs, limit=2)] == ["a__1", "b__2"]
        assert [r.sample_id for r in select_records(recs, datasets=["a"], limit=1)] == ["a__1"]
        assert select_records(recs, ids=["zzz"]) == []

    def test_realign_concat_segments(self):
        segs = [Segment("A", 3, 0.0, 10.0), Segment("B", 6, 10.0, 13.9), Segment("C", 0, 13.9, 20.81)]
        rec = _rec("ophnet__case_0002", "ophnet", source_kind="concat_clips", segments=segs)
        cmap = [{"clip": "c0", "offset_s": 0.0, "duration_s": 10.017}, {"clip": "c1", "offset_s": 10.017, "duration_s": 3.911},
                {"clip": "c2", "offset_s": 13.928, "duration_s": 6.947}]
        assert realign_concat_segments(rec, cmap) is True
        assert [s.start_s for s in rec.segments] == pytest.approx([0.0, 10.017, 13.928])
        assert rec.segments[-1].end_s == pytest.approx(20.875)
        assert rec.segments[1].extra["metadata_start_s"] == 10.0
        # mismatched counts or kinds -> untouched
        rec2 = _rec("x__1", "ophnet", source_kind="concat_clips", segments=[Segment("A", 1, 0.0, 10.0)])
        assert realign_concat_segments(rec2, cmap) is False

    def test_order_samples_fallback_interleaves(self):
        recs = [_rec(f"a__{i}", "a") for i in range(4)] + [_rec(f"b__{i}", "b") for i in range(2)] + [_rec("c__0", "c")]
        ordered = order_samples_fallback(recs)
        assert [r.sample_id for r in ordered] == ["a__0", "b__0", "a__1", "c__0", "a__2", "b__1", "a__3"]
        assert order_samples_fallback([]) == []
        # any prefix is dataset-mixed; within a dataset the manifest order is preserved
        assert [r.sample_id for r in ordered if r.dataset == "a"] == ["a__0", "a__1", "a__2", "a__3"]

    def test_canonical_order_and_select_batch(self):
        recs = [_rec(f"a__{i}", "a") for i in range(4)] + [_rec(f"b__{i}", "b") for i in range(2)]
        ordered = canonical_order(recs)
        assert len(ordered) == 6 and {r.sample_id for r in ordered} == {r.sample_id for r in recs}
        assert canonical_order(recs) == ordered  # deterministic
        b1, b2, b3, b4 = (select_batch(ordered, k, 2) for k in (1, 2, 3, 4))
        assert [r.sample_id for r in b1] + [r.sample_id for r in b2] + [r.sample_id for r in b3] == [r.sample_id for r in ordered]
        assert b4 == []
        assert {r.dataset for r in b1} == {"a", "b"}, "a batch must mix datasets"
        with pytest.raises(ValueError):
            select_batch(ordered, 0, 2)
        with pytest.raises(ValueError):
            select_batch(ordered, 1, 0)

    def test_timeline_note(self):
        probe = ProbeInfo(duration_s=20.83, fps=59.94, width=1920, height=1080)
        rec = _rec("ophnet__case_0002", "ophnet", source_kind="concat_clips",
                   segments=[Segment("A", 3, 0.0, 10.0)], labels={"original_segments": []})
        cmap = [{"clip": "c0", "offset_s": 0.0, "duration_s": 10.0}, {"clip": "c1", "offset_s": 10.0, "duration_s": 10.83}]
        note = build_timeline_note(rec, probe, cmap, realigned=True)
        assert "concatenat" in note and "clip 1 at 00:10.0" in note and "original_segments" in note
        plain = build_timeline_note(_rec("ophora__abc_1", "ophora"), ProbeInfo(duration_s=6.0))
        assert "No time-localised annotations" in plain and "Ophora" in plain


# ------------------------------------------------------------------------------------ sources (offline, synthetic clips)
import logging  # noqa: E402
import zipfile  # noqa: E402

from bench.config import DEFAULTS, REPO_ROOT, Config, deep_merge  # noqa: E402
from bench.util import CmdResult, make_zip_spec, run_cmd  # noqa: E402


def _synth_clip(path: Path, seconds: float, cfg: Config, size: str = "64x48", rate: int = 10) -> Path:
    cmd = [cfg.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi",
           "-i", f"testsrc=duration={seconds}:size={size}:rate={rate}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    res = run_cmd(cmd, timeout=120)
    assert res.ok, res.err
    return path


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def tmp_cfg(tmp_path, monkeypatch) -> Config:
    monkeypatch.setenv("OPHBENCH_DATA_DIR", str(tmp_path / "data"))
    cfg = Config(deep_merge(DEFAULTS, {}), root=REPO_ROOT)
    cfg.ensure_dirs()
    return cfg


class TestSourcesOffline:
    def test_materialize_file_symlink_idempotent(self, tmp_path, tmp_cfg):
        clip = _synth_clip(tmp_path / "clip.mp4", 1.0, tmp_cfg)
        rec = _rec("ophora__clip_1", "ophora", source_kind="file", source_paths=[str(clip)])
        dst = media.materialize_source(rec, tmp_path / "s", tmp_cfg)
        assert dst.name == "source.mp4" and dst.is_symlink() and dst.resolve() == clip.resolve()
        assert media.materialize_source(rec, tmp_path / "s", tmp_cfg) == dst  # second call: link kept
        rec_missing = _rec("ophora__nope", "ophora", source_kind="file", source_paths=[str(tmp_path / "nope.mp4")])
        with pytest.raises(media.MediaError):
            media.materialize_source(rec_missing, tmp_path / "s2", tmp_cfg)
        with pytest.raises(media.MediaError):
            media.materialize_source(_rec("x__1", "x", source_kind="weird", source_paths=["a"]), tmp_path / "s3", tmp_cfg)

    def test_extract_zip_member_cache_and_materialize(self, tmp_path, tmp_cfg):
        clip = _synth_clip(tmp_path / "clip.mp4", 1.0, tmp_cfg)
        zp = tmp_path / "videos_part001.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(clip, "sub/SK_0001.mp4")
        spec = make_zip_spec(zp, "sub/SK_0001.mp4")
        cache_root = tmp_path / "cache"
        out = media.extract_zip_member(spec, cache_root)
        assert out == cache_root / "sub" / "SK_0001.mp4"
        assert out.stat().st_size == clip.stat().st_size
        mtime = out.stat().st_mtime_ns
        assert media.extract_zip_member(spec, cache_root) == out
        assert out.stat().st_mtime_ns == mtime, "cached copy must not be rewritten"
        with pytest.raises(media.MediaError):
            media.extract_zip_member(make_zip_spec(zp, "missing.mp4"), cache_root)
        rec = _rec("lmm_skill__SK_0001", "lmm_skill", source_kind="zip_member", source_paths=[spec])
        dst = media.materialize_source(rec, tmp_path / "s", tmp_cfg)
        expected = Path(tmp_cfg.paths.cache) / "extracted" / "lmm_skill" / "sub" / "SK_0001.mp4"
        assert dst.is_symlink() and dst.resolve() == expected.resolve() and expected.exists()

    def test_concat_clips_copy_map_and_reuse(self, tmp_path, tmp_cfg):
        a = _synth_clip(tmp_path / "a.mp4", 2.0, tmp_cfg)
        b = _synth_clip(tmp_path / "b.mp4", 3.0, tmp_cfg)
        sdir = tmp_path / "s"
        out = media.concat_clips([a, b], sdir, tmp_cfg)
        assert out == sdir / "source.mp4" and out.exists() and not out.is_symlink()
        info = media.ffprobe(out, tmp_cfg)
        assert info.duration_s == pytest.approx(5.0, abs=0.3)
        cmap = __import__("json").load(open(sdir / "concat_map.json"))
        assert [e["offset_s"] for e in cmap] == pytest.approx([0.0, 2.0], abs=0.05)
        assert cmap[1]["offset_s"] == pytest.approx(cmap[0]["duration_s"])
        assert (sdir / "concat_list.txt").read_text().count("file '") == 2
        mtime = out.stat().st_mtime_ns
        media.concat_clips([a, b], sdir, tmp_cfg)
        assert out.stat().st_mtime_ns == mtime, "valid existing concat output must be reused"
        with pytest.raises(media.MediaError):
            media.concat_clips([a, tmp_path / "missing.mp4"], tmp_path / "s2", tmp_cfg)

    def test_concat_clips_falls_back_to_reencode(self, tmp_path, tmp_cfg, monkeypatch):
        a = _synth_clip(tmp_path / "a.mp4", 2.0, tmp_cfg)
        b = _synth_clip(tmp_path / "b.mp4", 2.0, tmp_cfg)
        real = media.run_ffmpeg

        def flaky(cmd, log=None, timeout=None):
            if "copy" in cmd:
                return CmdResult(1, "", "simulated stream-copy failure", cmd)
            return real(cmd, log, timeout)

        monkeypatch.setattr(media, "run_ffmpeg", flaky)
        lg = logging.Logger("t")
        h = _ListHandler()
        lg.addHandler(h)
        out = media.concat_clips([a, b], tmp_path / "s", tmp_cfg, lg, force=True)
        assert media.ffprobe(out, tmp_cfg).duration_s == pytest.approx(4.0, abs=0.3)
        assert any("falling back to re-encode" in m for m in h.messages)
        assert any("libx264" in m for m in h.messages), "the exact re-encode command must be logged"

    def test_extract_frame_and_past_eof(self, tmp_path, tmp_cfg):
        clip = _synth_clip(tmp_path / "clip.mp4", 2.0, tmp_cfg, size="320x240")
        out = media.extract_frame(clip, 1.0, tmp_path / "f.jpg", tmp_cfg, long_side=160)
        with Image.open(out) as im:
            assert im.size == (160, 120)
        out2 = media.extract_frame(clip, 1.0, tmp_path / "g.jpg", tmp_cfg, long_side=768)
        with Image.open(out2) as im:
            assert im.size == (320, 240), "never upscale"
        with pytest.raises(media.MediaError):
            media.extract_frame(clip, 100.0, tmp_path / "h.jpg", tmp_cfg)

    def test_make_preview(self, tmp_path, tmp_cfg):
        clip = _synth_clip(tmp_path / "clip.mp4", 2.0, tmp_cfg, size="320x240")
        out = media.make_preview(clip, tmp_path / "preview.mp4", height=120, crf=30, cfg=tmp_cfg)
        info = media.ffprobe(out, tmp_cfg)
        assert (info.width, info.height) == (160, 120)
        assert info.duration_s == pytest.approx(2.0, abs=0.2)
        capped = media.make_preview(clip, tmp_path / "p2.mp4", height=480, crf=30, max_duration_s=1.0, cfg=tmp_cfg)
        info2 = media.ffprobe(capped, tmp_cfg)
        assert info2.height == 240, "never upscale beyond source height"
        assert info2.duration_s == pytest.approx(1.0, abs=0.2)
