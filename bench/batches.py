"""Canonical benchmark order and annotation batches (DESIGN.md section 13.1).

Clinicians review the sampled videos in batches of about 100.  So that every batch spans all
datasets in their quota proportions, the manifest is put into one deterministic *benchmark order*:
within a dataset the manifest order is kept (the sampler sorts by ``sample_id``) and item ``i`` of
``n`` gets the position ``(i + 0.5) / n``; the global order is ``(position, dataset, sample_id)``.
Batch ``K`` (1-based) is the slice ``[(K-1)*size, K*size)`` of that order.

The order is always re-derived from the manifest rows (never read back from ``sample_index``), so
every stage agrees even for manifests written before this module existed; the sampler stores the
rank in ``SampleRecord.sample_index`` for convenience.  Records may be ``SampleRecord`` objects or
plain manifest dicts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

from bench.config import Config

DEFAULT_BATCH_SIZE = 100
PACKAGE_KINDS: tuple[str, ...] = ("preview", "source")
PACKAGES_DIR_ENV = "OPHBENCH_PACKAGES_DIR"


def _field(rec: Any, key: str) -> Any:
    return rec.get(key) if isinstance(rec, dict) else getattr(rec, key, None)


def sample_id_of(rec: Any) -> str:
    return str(_field(rec, "sample_id") or "")


def order_samples(records: Sequence[Any]) -> list[Any]:
    """Proportional interleave across datasets (within a dataset the given order is kept)."""
    per_ds: dict[str, list[Any]] = {}
    for r in records:
        per_ds.setdefault(str(_field(r, "dataset") or ""), []).append(r)
    keyed: list[tuple[float, str, str, Any]] = []
    for ds, items in per_ds.items():
        n = len(items)
        for i, r in enumerate(items):
            keyed.append(((i + 0.5) / n, ds, sample_id_of(r), r))
    keyed.sort(key=lambda k: (k[0], k[1], k[2]))
    return [k[3] for k in keyed]


def batch_size(cfg: Optional[Config], override: Optional[int] = None) -> int:
    """``override`` when given, else ``config: batches.size`` (default 100); always >= 1."""
    if override:
        return max(1, int(override))
    raw = cfg.get("batches.size", DEFAULT_BATCH_SIZE) if cfg is not None else DEFAULT_BATCH_SIZE
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_BATCH_SIZE


def n_batches(n_items: int, size: int) -> int:
    return (int(n_items) + size - 1) // size if n_items > 0 and size > 0 else 0


def batch_of(rank: int, size: int) -> int:
    """1-based batch number of the item at 0-based ``rank``."""
    return int(rank) // max(1, int(size)) + 1


def select_batch(ordered: Sequence[Any], batch: int, size: int) -> list[Any]:
    """Items ``[(batch-1)*size, batch*size)`` of ``ordered`` (1-based batch numbers)."""
    batch, size = int(batch), int(size)
    if batch < 1 or size < 1:
        raise ValueError(f"batch and batch size must be >= 1 (got batch={batch}, size={size})")
    return list(ordered)[(batch - 1) * size: batch * size]


def ranks(records: Sequence[Any]) -> dict[str, int]:
    """sample_id -> 0-based rank in the benchmark order."""
    return {sample_id_of(r): i for i, r in enumerate(order_samples(records))}


def assign_batches(records: Sequence[Any], size: int) -> dict[str, int]:
    """sample_id -> 1-based batch number."""
    return {sid: batch_of(rank, size) for sid, rank in ranks(records).items()}


def batch_dir_name(batch: int) -> str:
    return f"batch_{int(batch):03d}"


def package_kind(cfg: Optional[Config], override: Optional[str] = None) -> str:
    """``preview`` (360p previews, default) or ``source`` (original / concatenated case video)."""
    kind = str(override or (cfg.get("batches.package_kind", "preview") if cfg is not None else "preview") or "preview")
    if kind not in PACKAGE_KINDS:
        raise ValueError(f"package kind must be one of {PACKAGE_KINDS}, got {kind!r}")
    return kind


def packages_dir(cfg: Config, override: Optional[str] = None) -> Path:
    """Where offline packages go: ``--packages-dir``, else $OPHBENCH_PACKAGES_DIR, else ``<repo>/ui_packages``."""
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get(PACKAGES_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(cfg.root) / "ui_packages"


def describe(records: Sequence[Any], size: int) -> list[dict]:
    """Per-batch summary rows ``[{batch, n, first_rank, datasets: {ds: n}}]`` (for status / reports)."""
    out: list[dict] = []
    ordered = order_samples(records)
    for b in range(1, n_batches(len(ordered), size) + 1):
        items = select_batch(ordered, b, size)
        ds: dict[str, int] = {}
        for r in items:
            d = str(_field(r, "dataset") or "")
            ds[d] = ds.get(d, 0) + 1
        out.append({"batch": b, "n": len(items), "first_rank": (b - 1) * size, "datasets": dict(sorted(ds.items()))})
    return out


# ------------------------------------------------------------------------------------------------
# Per-batch layout of data/prepared:  data/prepared/batch_<KKK>/<sample_id>/
# ------------------------------------------------------------------------------------------------
import json as _json
import threading as _threading

_BATCH_MAP_CACHE: dict[tuple, dict[str, int]] = {}
_BATCH_MAP_LOCK = _threading.Lock()


def manifest_batch_map(cfg: Config) -> dict[str, int]:
    """sample_id -> 1-based batch number from the current sample manifest (config batch size).

    Uses ``config: batches.size`` (not a ``--batch-size`` override) so folder names stay stable.
    Cached per manifest file (path, mtime, size); empty dict when there is no manifest.
    """
    path = Path(cfg.paths.sample) / "sample_manifest.jsonl"
    try:
        st = path.stat()
    except OSError:
        return {}
    size = batch_size(cfg)
    key = (str(path), st.st_mtime_ns, st.st_size, size)
    with _BATCH_MAP_LOCK:
        cached = _BATCH_MAP_CACHE.get(key)
        if cached is not None:
            return cached
        rows: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = _json.loads(line)
                except ValueError:
                    continue
                rows.append({"sample_id": d.get("sample_id"), "dataset": d.get("dataset")})
        mapping = {sid: batch_of(rank, size) for sid, rank in ranks(rows).items()}
        _BATCH_MAP_CACHE.clear()
        _BATCH_MAP_CACHE[key] = mapping
        return mapping


def prepared_dir(cfg: Config, sample_id: str, *, for_write: bool = False) -> Path:
    """Folder of a prepared sample: ``data/prepared/batch_<KKK>/<sample_id>``.

    Samples not in the manifest (e.g. synthetic test data) use the flat legacy layout
    ``data/prepared/batch_<KKK>/<sample_id>``.  When reading (``for_write=False``) an existing legacy flat
    folder is still found, so older data keeps working.
    """
    root = Path(cfg.paths.prepared)
    b = manifest_batch_map(cfg).get(sample_id)
    batched = root / batch_dir_name(b) / sample_id if b else None
    flat = root / sample_id
    if for_write:
        return batched or flat
    if batched is not None and batched.exists():
        return batched
    if flat.exists():
        return flat
    return batched or flat
