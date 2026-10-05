"""Offline tests for the sampling stage (bench.sampler) and the pure helpers of bench.inventory.

Synthetic VideoRecords are written as inventories into a temporary data dir selected through the
OPHBENCH_DATA_DIR / OPHBENCH_LOGS_DIR environment variables; no dataset files or network access
are needed.
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import pytest
import yaml

from bench import inventory, sampler
from bench.config import load_config
from bench.schema import DATASETS, SampleRecord, Segment, VideoRecord, make_sample_id
from bench.util import read_jsonl

QUOTAS = {"cataract101": 6, "cataract1k": 10, "lmm_phase": 4, "lmm_skill": 8, "lmm_raw": 6, "migs": 6,
          "ophnet": 10, "ophora": 10}
TARGET = sum(QUOTAS.values())


# ----------------------------------------------------------------------------- synthetic inventories
def mk(dataset: str, vid: str, duration=600.0, strata=None, labels=None, segments=None, **kw) -> VideoRecord:
    kw.setdefault("source_kind", "file")
    kw.setdefault("source_paths", [f"/nonexistent/{dataset}/{vid}.mp4"])
    kw.setdefault("procedure_category", "cataract")
    return VideoRecord(sample_id=make_sample_id(dataset, vid), dataset=dataset, video_id=str(vid),
                       duration_s=duration, strata=dict(strata or {}), labels=dict(labels or {}),
                       segments=list(segments or []), **kw)


def _phases(n: int, total: float) -> list[Segment]:
    step = total / max(1, n)
    return [Segment(label=f"phase{i}", label_id=i, start_s=i * step, end_s=(i + 1) * step) for i in range(n)]


def synth_inventories() -> dict[str, list[VideoRecord]]:
    inv: dict[str, list[VideoRecord]] = {d: [] for d in DATASETS}
    for i in range(12):
        inv["cataract101"].append(mk("cataract101", f"case_{i}", 240 + i * 60,
                                     strata={"experience": "low" if i % 2 else "high", "surgeon": (i % 4) + 1},
                                     labels={"surgeon_id": (i % 4) + 1}))
    for i in range(30):
        if i < 12:
            d = 400.0 + i * 40
            flags = ["Iris Hooks"] if i < 3 else []
            inv["cataract1k"].append(mk("cataract1k", f"case_{2000 + i}", d, segments=_phases(5, d),
                                        strata={"annotated": True, "rare_flag": bool(flags)},
                                        labels={"annotated": True, "flags": flags}))
        else:
            inv["cataract1k"].append(mk("cataract1k", f"case_{2000 + i}", None,
                                        strata={"annotated": False, "rare_flag": False,
                                                "size_bin": "small" if i % 2 else "large"},
                                        labels={"annotated": False, "file_size_mb": 100 + i}))
    for i in range(8):
        site = "S1" if i < 5 else "S2"
        inv["lmm_phase"].append(mk("lmm_phase", f"PH_{i:04d}_{100 + i:04d}_{site}", 500 + i * 80,
                                   source_kind="zip_member", source_paths=[f"/nonexistent/z.zip!PH_{i}.mp4"],
                                   strata={"site": site, "idle_share_bin": "<0.2"},
                                   labels={"site": site, "idle_share": 0.1}))
    for i in range(20):
        site = "S1" if i % 3 else "S2"
        inv["lmm_skill"].append(mk("lmm_skill", f"SK_{i:04d}_{site}_P03", 90.0,
                                   strata={"skill_tertile": "low" if i < 7 else ("mid" if i < 14 else "high"),
                                           "site": site, "adverse_event": 1 if i < 6 else 0},
                                   labels={"adverse_event": 1 if i < 6 else 0, "averaged": 2.0 + i * 0.15,
                                           "site": site}))
    for i in range(40):
        s2 = 30 <= i < 38
        site = "S2" if s2 else "S1"
        if s2:
            d = 200.0 if i < 34 else 600.0
        elif i < 15:
            d = 100.0
        else:
            d = 300.0 + (i - 15) * 70
        inv["lmm_raw"].append(mk("lmm_raw", f"RV_{i:04d}_{site}", d,
                                 strata={"site": site}, labels={"site": site, "in_phase_subset": i < 10}))
    ops = ["GT120", "GT240", "GT360", "PEI", "GATT"]
    for i in range(15):
        inv["migs"].append(mk("migs", str(i + 1), 300 + i * 30, procedure_category="glaucoma",
                              strata={"operation_type": ops[i % 5], "knife": "3.2mm" if i % 2 else "15deg"},
                              labels={"operation_type": ops[i % 5], "knife": "3.2mm" if i % 2 else "15deg"}))
    surgeries = ["Phaco", "Trabeculectomy", "DMEK", "Pterygium", "ICL"]
    cats = ["cataract", "glaucoma", "cornea", "cornea", "refractive"]
    for i in range(30):
        n_ph = (i % 8) + 1
        d = [30.0, 90.0, 400.0, 900.0, 1500.0][(i // 5) % 5]  # decorrelated from the surgery (i % 5)
        inv["ophnet"].append(mk("ophnet", f"case_{i:04d}", d, segments=_phases(n_ph, d), source_kind="concat_clips",
                                source_paths=[f"/nonexistent/ophnet/case_{i:04d}_{k}.mp4" for k in range(n_ph)],
                                procedure_category=cats[i % 5],
                                strata={"primary_surgery": surgeries[i % 5], "n_phases_bin": str(n_ph)},
                                labels={"primary_surgery": surgeries[i % 5], "n_phases": n_ph}))
    ocats = ["cataract", "glaucoma", "retina", "cornea"]
    for s in range(12):
        src = f"src{s:02d}"
        for k in range(3):
            words = 15 + k * 4
            inv["ophora"].append(mk("ophora", f"{src}_{k}", None, procedure_category=ocats[s % 4],
                                    labels={"instruction": " ".join(["word"] * words), "instruction_words": words,
                                            "source_video_id": src, "clip_index": k}))
    for s in range(2):  # too-short instructions -> excluded by min_instruction_words
        inv["ophora"].append(mk("ophora", f"short{s}_0", None, procedure_category="cataract",
                                labels={"instruction": "short clip here", "instruction_words": 3,
                                        "source_video_id": f"short{s}"}))
    return inv


# ----------------------------------------------------------------------------- fixtures / helpers
def make_cfg(tmp_path: Path, monkeypatch, quotas=QUOTAS, target=TARGET, seed=7, extra_sampling=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    monkeypatch.setenv("OPHBENCH_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPHBENCH_LOGS_DIR", str(tmp_path / "logs"))
    sampling = {"duration_bins_s": [300, 480, 720],
                "lmm_raw": {"min_duration_s": 300, "max_duration_s": 1500, "s2_min": 2, "exclude_phase_subset": True},
                "ophora": {"one_clip_per_source_video": True, "min_instruction_words": 12,
                           "category_weights": {"cataract": 1.0, "glaucoma": 2.0, "retina": 1.5, "cornea": 2.0}}}
    if extra_sampling:
        sampling.update(extra_sampling)
    raw = {"seed": seed, "data_dir": str(data_dir), "logs_dir": str(tmp_path / "logs"),
           "target_total": target, "quotas": quotas, "sampling": sampling}
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_config(cfg_path)
    cfg.ensure_dirs()
    return cfg


def write_inventories(cfg, inventories: dict[str, list[VideoRecord]]) -> None:
    for name, recs in inventories.items():
        with open(Path(cfg.paths.inventory) / f"{name}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r.to_dict()) + "\n")


def run_sample(cfg, argv: list[str] | None = None) -> tuple[int, list[SampleRecord], dict]:
    p = argparse.ArgumentParser()
    sampler.add_args(p)
    args = p.parse_args(argv or [])
    rc = sampler.main(args, cfg)
    mpath = Path(cfg.paths.sample) / "sample_manifest.jsonl"
    manifest = [SampleRecord.from_dict(d) for d in read_jsonl(mpath)] if mpath.exists() else []
    rpath = Path(cfg.paths.sample) / "sampling_report.json"
    report = json.loads(rpath.read_text()) if rpath.exists() else {}
    return rc, manifest, report


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = make_cfg(tmp_path, monkeypatch)
    write_inventories(c, synth_inventories())
    return c


def by_dataset(manifest: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
    out: dict[str, list[SampleRecord]] = {}
    for m in manifest:
        out.setdefault(m.dataset, []).append(m)
    return out


# ----------------------------------------------------------------------------- allocator
@pytest.mark.parametrize("seed", range(6))
def test_allocate_invariants(seed):
    rng = random.Random(seed)
    sizes = {f"s{i}": rng.choice([0, 1, 2, 5, 17, 60, 300]) for i in range(rng.randint(1, 12))}
    total = sum(sizes.values())
    nonempty = [k for k, v in sizes.items() if v > 0]
    for quota in (0, 1, 3, 7, 25, 100, total, total + 50):
        alloc = sampler.allocate(sizes, quota, random.Random(seed))
        assert set(alloc) == set(sizes)
        assert sum(alloc.values()) == min(quota, total)
        assert all(0 <= alloc[k] <= sizes[k] for k in sizes)
        if quota >= len(nonempty):
            assert all(alloc[k] >= 1 for k in nonempty), (sizes, quota, alloc)
        assert all(alloc[k] == 0 for k in sizes if sizes[k] == 0)


def test_allocate_sqrt_proportional_and_floor():
    alloc = sampler.allocate({"big": 400, "mid": 100, "tiny": 1, "small": 4}, 30, random.Random(0))
    assert sum(alloc.values()) == 30
    assert alloc["tiny"] == 1 and alloc["small"] >= 1
    # sqrt(400)=20 vs sqrt(100)=10: big should get about twice mid, not four times
    assert 1.5 <= alloc["big"] / alloc["mid"] <= 2.6, alloc


def test_allocate_weights_shift_shares():
    sizes = {"cataract": 900, "glaucoma": 100}
    plain = sampler.allocate(sizes, 50, random.Random(0))
    weighted = sampler.allocate(sizes, 50, random.Random(0), weights={"cataract": 1.0, "glaucoma": 3.0})
    assert sum(weighted.values()) == 50
    assert weighted["glaucoma"] > plain["glaucoma"]


def test_allocate_quota_exceeds_pool_takes_all():
    assert sampler.allocate({"a": 3, "b": 0, "c": 2}, 99, random.Random(0)) == {"a": 3, "b": 0, "c": 2}
    assert sampler.allocate({}, 5, random.Random(0)) == {}
    assert sampler.allocate({"a": 5}, 0, random.Random(0)) == {"a": 0}


def test_scale_quotas_sums_to_target():
    q = sampler.scale_quotas({"a": 101, "b": 300, "c": 420, "d": 0}, 100, random.Random(0))
    assert sum(q.values()) == 100 and q["d"] == 0 and q["c"] > q["b"] > q["a"] > 0


# ----------------------------------------------------------------------------- end-to-end sampling
def test_total_equals_target_and_outputs(cfg):
    rc, manifest, report = run_sample(cfg)
    assert rc == 0
    assert len(manifest) == TARGET
    per = {ds: len(v) for ds, v in by_dataset(manifest).items()}
    assert per == QUOTAS
    assert [m.sample_index for m in manifest] == list(range(TARGET))
    assert len({m.sample_id for m in manifest}) == TARGET
    assert all(m.stratum_key and m.selection_reason for m in manifest)
    sd = Path(cfg.paths.sample)
    csv_lines = (sd / "sample_manifest.csv").read_text().splitlines()
    assert len(csv_lines) == TARGET + 1
    assert csv_lines[0] == ",".join(sampler.MANIFEST_CSV_FIELDS)
    assert (sd / "sampling_report.md").exists()
    assert report["total_selected"] == TARGET and report["redistribution_log"] == []
    for ds in QUOTAS:
        d = report["datasets"][ds]
        assert d["selected"] == QUOTAS[ds] and d["shortfall"] == 0
        assert sum(s["selected"] for s in d["strata"]) == d["selected"]
        assert all(s["selected"] <= s["pool"] for s in d["strata"])
    assert (Path(cfg.paths.logs) / "runs.jsonl").exists()


def test_redistribution_when_dataset_pool_is_short(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    inv = synth_inventories()
    inv["cataract101"] = inv["cataract101"][:2]  # quota 6 but only 2 videos
    write_inventories(cfg, inv)
    rc, manifest, report = run_sample(cfg)
    assert rc == 0
    assert len(manifest) == TARGET
    per = {ds: len(v) for ds, v in by_dataset(manifest).items()}
    assert per["cataract101"] == 2
    assert sum(per.values()) == TARGET
    assert report["datasets"]["cataract101"]["shortfall"] == 4
    assert len(report["redistribution_log"]) >= 1
    assert sum(d["topup"] for d in report["datasets"].values()) == 4
    assert any("redistribution" in m.selection_reason for m in manifest)


def test_missing_inventory_is_redistributed(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    inv = synth_inventories()
    del inv["migs"]
    write_inventories(cfg, inv)
    rc, manifest, report = run_sample(cfg)
    assert rc == 0 and len(manifest) == TARGET
    assert "migs" not in by_dataset(manifest)
    assert report["missing_inventories"] == ["migs"]


def test_pools_exhausted_returns_fewer_without_crash(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch, quotas={"cataract101": 5, "migs": 5}, target=10)
    inv = synth_inventories()
    write_inventories(cfg, {"cataract101": inv["cataract101"][:3], "migs": inv["migs"][:2]})
    rc, manifest, report = run_sample(cfg)
    assert rc == 0 and len(manifest) == 5
    assert report["redistribution_log"][-1].get("note") == "all eligible pools exhausted"


def test_ophora_one_clip_per_source_and_min_words(cfg):
    rc, manifest, _ = run_sample(cfg)
    oph = by_dataset(manifest)["ophora"]
    assert len(oph) == QUOTAS["ophora"]
    sources = [m.labels["source_video_id"] for m in oph]
    assert len(set(sources)) == len(sources)
    assert not any(s.startswith("short") for s in sources)
    # within a source the longest instruction (k=2) is preferred
    assert all(m.labels["clip_index"] == 2 for m in oph)
    assert all(m.stratum_key == m.procedure_category for m in oph)


def test_lmm_skill_keeps_all_adverse_clips(cfg):
    rc, manifest, _ = run_sample(cfg)
    sk = by_dataset(manifest)["lmm_skill"]
    assert len(sk) == QUOTAS["lmm_skill"]
    adverse = [m for m in sk if m.labels["adverse_event"] == 1]
    assert len(adverse) == 6
    assert all("adverse-event clip" in m.selection_reason for m in adverse)
    assert len({m.stratum_key for m in sk}) >= 3  # balanced over tertile x site


def test_lmm_raw_rules(cfg):
    rc, manifest, report = run_sample(cfg)
    raw = by_dataset(manifest)["lmm_raw"]
    assert len(raw) == QUOTAS["lmm_raw"]
    assert not any(m.labels["in_phase_subset"] for m in raw)
    s2 = [m for m in raw if m.labels["site"] == "S2"]
    assert len(s2) >= 2  # s2_min
    assert all(300 <= m.duration_s <= 1500 for m in raw)
    assert report["datasets"]["lmm_raw"]["excluded"]


def test_migs_every_operation_type_represented(cfg):
    rc, manifest, _ = run_sample(cfg)
    migs = by_dataset(manifest)["migs"]
    assert len(migs) == QUOTAS["migs"]
    assert {m.labels["operation_type"] for m in migs} == {"GT120", "GT240", "GT360", "PEI", "GATT"}


def test_ophnet_prefers_rich_cases_and_covers_surgeries(cfg):
    rc, manifest, _ = run_sample(cfg)
    oph = by_dataset(manifest)["ophnet"]
    assert len(oph) == QUOTAS["ophnet"]
    assert {m.labels["primary_surgery"] for m in oph} == {"Phaco", "Trabeculectomy", "DMEK", "Pterygium", "ICL"}
    preferred = [m for m in oph if m.labels["n_phases"] >= 3 and 60 <= m.duration_s <= 1200]
    assert len(preferred) == 10  # every surgery has >= 2 preferred cases, and those are filled first


def test_cataract1k_rare_first_and_annotated_share(cfg):
    rc, manifest, report = run_sample(cfg)
    c1k = by_dataset(manifest)["cataract1k"]
    assert len(c1k) == QUOTAS["cataract1k"]
    rare = [m for m in c1k if m.labels.get("flags")]
    assert len(rare) == 3
    annotated = [m for m in c1k if m.labels["annotated"]]
    assert len(annotated) == round(QUOTAS["cataract1k"] * 0.87)
    groups = {g["name"]: g for g in report["datasets"]["cataract1k"]["groups"]}
    assert groups["annotated"]["must_include"] == 3
    assert all(m.stratum_key.startswith("unannotated|size:") for m in c1k if not m.labels["annotated"])


def test_missing_fields_degrade_gracefully(tmp_path, monkeypatch):
    quotas = {d: 3 for d in DATASETS}
    cfg = make_cfg(tmp_path, monkeypatch, quotas=quotas, target=sum(quotas.values()))
    bare = {d: [mk(d, f"v{i}", None) for i in range(5)] for d in DATASETS}
    for r in bare["lmm_raw"]:
        r.labels["site"] = "S2"  # S2 videos stay eligible even without a known duration
    write_inventories(cfg, bare)
    rc, manifest, report = run_sample(cfg)
    assert rc == 0
    assert len(manifest) == sum(quotas.values())
    assert all(m.stratum_key for m in manifest)
    assert any("unknown" in m.stratum_key for m in manifest)


def test_determinism_across_runs(tmp_path, monkeypatch):
    runs = []
    for i in range(2):
        cfg = make_cfg(tmp_path / f"run{i}", monkeypatch)
        write_inventories(cfg, synth_inventories())
        rc, manifest, _ = run_sample(cfg, ["--seed", "42"])
        assert rc == 0
        runs.append([(m.sample_id, m.stratum_key, m.selection_reason) for m in manifest])
    assert runs[0] == runs[1]
    cfg = make_cfg(tmp_path / "run_other_seed", monkeypatch)
    write_inventories(cfg, synth_inventories())
    _, other, _ = run_sample(cfg, ["--seed", "43"])
    assert len(other) == len(runs[0])


def test_target_override_scales_quotas(cfg):
    rc, manifest, report = run_sample(cfg, ["--target", "30"])
    assert rc == 0 and len(manifest) == 30
    assert sum(report["quotas_effective"].values()) == 30
    assert report["quotas_configured"] == QUOTAS


def test_dry_run_writes_no_manifest(cfg, capsys):
    rc, manifest, report = run_sample(cfg, ["--dry-run"])
    assert rc == 0 and manifest == [] and report == {}
    assert not (Path(cfg.paths.sample) / "sample_manifest.csv").exists()
    out = capsys.readouterr().out
    assert "# Sampling report" in out and "| dataset |" in out


def test_no_inventory_returns_failure(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    rc, manifest, _ = run_sample(cfg)
    assert rc == 1 and manifest == []


# ----------------------------------------------------------------------------- inventory helpers (offline)
def test_parse_rate_and_ffprobe_json():
    assert inventory.parse_rate("30000/1001") == 29.97
    assert inventory.parse_rate("60/1") == 60.0
    assert inventory.parse_rate("0/0") is None and inventory.parse_rate(None) is None
    res = inventory.parse_ffprobe_json(json.dumps({"streams": [{"width": 640, "height": 360, "r_frame_rate": "60/1"}],
                                                   "format": {"duration": "712.600000"}}))
    assert res == {"ok": True, "duration_s": 712.6, "fps": 60.0, "width": 640, "height": 360, "error": None,
                   "tool": "ffprobe"}
    assert inventory.parse_ffprobe_json("{}")["ok"] is False


def test_parse_ffmpeg_stderr():
    text = ("Input #0, mov,mp4, from 'x.mp4':\n  Duration: 00:11:52.60, start: 0.000000, bitrate: 1178 kb/s\n"
            "    Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709), 640x360 "
            "[SAR 1:1 DAR 16:9], 1175 kb/s, 60 fps, 60 tbr, 90k tbn, 120 tbc (default)\n")
    res = inventory.parse_ffmpeg_stderr(text)
    assert res["ok"] and res["duration_s"] == 712.6 and (res["width"], res["height"]) == (640, 360)
    assert res["fps"] == 60.0
    assert inventory.parse_ffmpeg_stderr("garbage")["ok"] is False


def test_path_checker_and_verify_record(tmp_path):
    import logging
    d = tmp_path / "clips"
    d.mkdir()
    for i in range(10):
        (d / f"c{i}.mp4").write_bytes(b"x")
    zp = tmp_path / "a.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("inner/m1.mp4", b"x")
        zf.writestr("m2.mp4", b"x")
    chk = inventory.PathChecker(logging.getLogger("t"))
    paths = [str(d / f"c{i}.mp4") for i in range(10)] + [str(d / "nope.mp4"), str(tmp_path / "lonely.mp4")]
    chk.prime(paths)
    assert chk.file_exists(str(d / "c3.mp4")) and not chk.file_exists(str(d / "nope.mp4"))
    assert not chk.file_exists(str(tmp_path / "lonely.mp4"))
    assert chk.n_scandir == 1 and chk.n_stat == 1  # hot dir listed once; cold path stat()ed
    assert chk.zip_member_exists(f"{zp}!m2.mp4") == (True, "")
    ok, why = chk.zip_member_exists(f"{zp}!m1.mp4")
    assert not ok and "found as 'inner/m1.mp4'" in why
    assert not chk.zip_member_exists(f"{tmp_path / 'missing.zip'}!m.mp4")[0]
    assert chk.n_zip_opened == 1

    rec_ok = mk("migs", "1", source_paths=[str(d / "c1.mp4")])
    rec_zip = mk("lmm_phase", "p", source_kind="zip_member", source_paths=[f"{zp}!m2.mp4"])
    rec_bad = mk("ophnet", "c", source_kind="concat_clips", source_paths=[str(d / "c1.mp4"), str(d / "zz.mp4")])
    rec_none = mk("ophora", "o", source_paths=[])
    assert inventory.verify_record(rec_ok, chk) == []
    assert inventory.verify_record(rec_zip, chk) == []
    assert inventory.verify_record(rec_bad, chk)[0].startswith("missing 1/2 clips")
    assert inventory.verify_record(rec_none, chk) == ["no source_paths"]


def test_probe_cache_roundtrip_and_apply(tmp_path):
    cache = inventory.ProbeCache(tmp_path / "probe_cache.jsonl")
    key = inventory.ProbeCache.key("/x/a.mp4", 123)
    assert cache.get(key) is None
    res = {"ok": True, "duration_s": 333.0, "fps": 29.97, "width": 720, "height": 480, "error": None, "tool": "ffprobe"}
    cache.put(key, "/x/a.mp4", 123, res)
    again = inventory.ProbeCache(tmp_path / "probe_cache.jsonl")
    assert again.get(key)["duration_s"] == 333.0
    rec = mk("cataract1k", "case_1", None, strata={"size_bin": "small"})
    inventory.apply_probe(rec, again.get(key), [300, 480, 720])
    assert rec.duration_s == 333.0 and rec.fps == 29.97 and rec.width == 720
    assert rec.strata["duration_bin"] == "5-8min" and rec.labels["duration_probed"] is True


def test_label_histograms_and_dataset_list():
    recs = [mk("migs", str(i), labels={"knife": "a" if i % 2 else "b", "frames": i * 1000, "flags": ["x"] if i < 2 else []})
            for i in range(30)]
    hist = inventory.label_histograms(recs)
    assert hist["knife"] == {"a": 15, "b": 15}
    assert "frames" not in hist  # high cardinality dropped
    assert hist["flags"] == {"x": 2}
    assert inventory.parse_dataset_list(" migs, ophnet,,migs ") == ["migs", "ophnet"]
