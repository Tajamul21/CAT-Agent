"""data/prepared/batch_<KKK>/<sample_id>/ layout resolved by bench.batches.prepared_dir."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.batches import manifest_batch_map, prepared_dir  # noqa: E402
from bench.config import load_config  # noqa: E402


def _cfg(tmp_path, monkeypatch, n=5, size=2):
    monkeypatch.setenv("OPHBENCH_DATA_DIR", str(tmp_path / "data"))
    cfg = load_config()
    cfg.raw.setdefault("batches", {})["size"] = size
    (tmp_path / "data" / "sample").mkdir(parents=True)
    rows = [{"sample_id": f"ds__v{i}", "dataset": "ds"} for i in range(n)]
    (tmp_path / "data" / "sample" / "sample_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return cfg


def test_batched_paths(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    m = manifest_batch_map(cfg)
    assert sorted(m.values()) == [1, 1, 2, 2, 3]
    root = Path(cfg.paths.prepared)
    sid = next(s for s, b in m.items() if b == 2)
    assert prepared_dir(cfg, sid, for_write=True) == root / "batch_002" / sid
    assert prepared_dir(cfg, sid) == root / "batch_002" / sid  # nothing on disk yet -> batched path


def test_legacy_flat_folder_still_found(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    root = Path(cfg.paths.prepared)
    (root / "ds__v0").mkdir(parents=True)
    assert prepared_dir(cfg, "ds__v0") == root / "ds__v0"
    assert prepared_dir(cfg, "ds__v0", for_write=True).parent.name.startswith("batch_")


def test_unknown_sample_uses_flat(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    assert prepared_dir(cfg, "other__x", for_write=True) == Path(cfg.paths.prepared) / "other__x"
