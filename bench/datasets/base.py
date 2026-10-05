"""Dataset adapter base class (DESIGN.md §3).

Adapters are pure metadata readers: they parse CSV/JSON/XLSX tables and zip tables of contents,
never decode video. Each yields ``VideoRecord`` objects for every candidate video in the dataset.
"""
from __future__ import annotations

import abc
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

from bench.config import Config
from bench.schema import DATASET_META, Segment, VideoRecord, make_sample_id
from bench.util import duration_bin


class DatasetAdapter(abc.ABC):
    #: registry key, one of schema.DATASETS
    name: str = ""

    def __init__(self, cfg: Config, log: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.root: Path = Path(cfg.paths.datasets_root)
        self.log = log or logging.getLogger(f"ophbench.inventory.{self.name}")
        self.bins = list(cfg.get("sampling.duration_bins_s", [300, 480, 720]))

    # ---- metadata -----------------------------------------------------------------------
    @property
    def meta(self) -> dict[str, str]:
        return DATASET_META.get(self.name, {})

    @property
    def title(self) -> str:
        return self.meta.get("title", self.name)

    @property
    def license(self) -> str:
        return self.meta.get("license", "")

    @property
    def citation(self) -> str:
        return self.meta.get("citation", "")

    @property
    def description(self) -> str:
        return self.meta.get("blurb", "")

    # ---- core ---------------------------------------------------------------------------
    @abc.abstractmethod
    def iter_records(self) -> Iterator[VideoRecord]:
        """Yield one VideoRecord per candidate video (all candidates, not only the sampled ones)."""

    def records(self) -> list[VideoRecord]:
        return list(self.iter_records())

    # ---- helpers for subclasses -----------------------------------------------------------
    def new_record(self, video_id: Any, **kw: Any) -> VideoRecord:
        kw.setdefault("license", self.license)
        kw.setdefault("citation", self.citation)
        return VideoRecord(sample_id=make_sample_id(self.name, video_id), dataset=self.name, video_id=str(video_id), **kw)

    def dbin(self, duration_s: Optional[float]) -> str:
        return duration_bin(duration_s, self.bins)

    @staticmethod
    def segments_from_starts(starts: list[tuple[float, str, Any]], total_s: float, kind: str = "phase") -> list[Segment]:
        """Build contiguous segments from (start_s, label, label_id) sorted by start; last ends at total_s."""
        starts = sorted(starts, key=lambda x: x[0])
        segs: list[Segment] = []
        for i, (s, label, lid) in enumerate(starts):
            e = starts[i + 1][0] if i + 1 < len(starts) else total_s
            if e < s:
                e = s
            segs.append(Segment(label=str(label), label_id=lid, start_s=float(s), end_s=float(e), kind=kind))
        return segs

    # ---- statistics ---------------------------------------------------------------------
    def stats(self, records: list[VideoRecord]) -> dict:
        durs = [r.duration_s for r in records if r.duration_s is not None]
        strata_hist: dict[str, dict[str, int]] = {}
        for r in records:
            for k, v in (r.strata or {}).items():
                strata_hist.setdefault(k, Counter())[str(v)] += 1  # type: ignore[index]
        return {
            "dataset": self.name,
            "title": self.title,
            "n_records": len(records),
            "n_with_duration": len(durs),
            "total_hours": round(sum(durs) / 3600.0, 2) if durs else None,
            "mean_duration_s": round(sum(durs) / len(durs), 1) if durs else None,
            "source_kinds": dict(Counter(r.source_kind for r in records)),
            "procedure_categories": dict(Counter(r.procedure_category for r in records)),
            "strata": {k: dict(sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))) for k, v in strata_hist.items()},
            "n_with_segments": sum(1 for r in records if r.segments),
            "notes": sorted(set(n for r in records for n in r.notes))[:50],
        }
