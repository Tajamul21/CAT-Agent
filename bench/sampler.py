"""Sampling stage (DESIGN.md section 5).

Reads ``data/inventory/<dataset>.jsonl``, builds every dataset's *eligible pool* with the
per-dataset rules of section 5.1, allocates the dataset quota over strata with :func:`allocate`
(section 5.2), redistributes shortfalls across datasets (section 5.3) and writes the manifest and
sampling report (section 5.4).

Everything is deterministic for a given seed: one ``random.Random(seed)`` is consumed in a fixed
order, every candidate list is sorted by ``sample_id`` before shuffling, and dict iteration is
always over sorted keys.

Usage: ``./ophbench sample [--target N] [--seed S] [--dry-run]``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from bench.batches import order_samples
from bench.config import Config
from bench.log import JsonlWriter, get_logger, now_iso, record_run
from bench.schema import DATASETS, SampleRecord, VideoRecord
from bench.util import duration_bin, fmt_duration, md_table, write_csv, write_json, write_jsonl

STAGE = "sample"
UNKNOWN = "unknown"
MANIFEST_CSV_FIELDS = ["sample_id", "dataset", "video_id", "procedure", "procedure_category", "duration_s",
                       "stratum_key", "selection_reason", "source_kind", "first_source_path"]

StratumFn = Callable[[VideoRecord], str]
PriorityFn = Callable[[VideoRecord], Any]
Describer = Callable[[VideoRecord], str]


# ======================================================================================= allocator
def _largest_remainder(weights: dict[str, float], caps: dict[str, int], total: int, rng: random.Random) -> dict[str, int]:
    """Distribute ``total`` units over keys proportionally to ``weights`` with largest-remainder
    rounding, never exceeding ``caps``; capped leftovers are redistributed until ``total`` is
    reached or every key is saturated. Ties in the remainder are broken with ``rng``."""
    out = {k: 0 for k in caps}
    remaining = max(0, int(total))
    while remaining > 0:
        open_keys = [k for k in sorted(caps) if caps[k] - out[k] > 0 and weights.get(k, 0.0) > 0]
        if not open_keys:
            break
        wsum = sum(weights[k] for k in open_keys)
        raw = {k: remaining * weights[k] / wsum for k in open_keys}
        base = {k: min(int(math.floor(raw[k])), caps[k] - out[k]) for k in open_keys}
        left = remaining - sum(base.values())
        order = sorted(open_keys, key=lambda k: (-(raw[k] - math.floor(raw[k])), rng.random()))
        for k in order:
            if left <= 0:
                break
            if base[k] < caps[k] - out[k]:
                base[k] += 1
                left -= 1
        for k in open_keys:
            out[k] += base[k]
        remaining = total - sum(out.values())
    return out


def allocate(strata_sizes: dict[str, int], quota: int, rng: Optional[random.Random] = None,
             weights: Optional[dict[str, float]] = None) -> dict[str, int]:
    """Allocate ``quota`` picks over strata (DESIGN.md 5.2).

    1. floor of 1 per non-empty stratum when ``quota >= number of non-empty strata``;
    2. the remainder proportional to ``weight * sqrt(size)`` (weights default to 1.0);
    3. largest-remainder rounding, capped by stratum size, leftovers redistributed.

    Returns a dict with every input key (empty strata get 0). The result sums to
    ``min(quota, sum(sizes))`` and never exceeds any stratum size.
    """
    rng = rng if rng is not None else random.Random(0)
    sizes = {str(k): max(0, int(v)) for k, v in strata_sizes.items()}
    alloc = {k: 0 for k in sizes}
    quota = max(0, int(quota))
    nonempty = sorted(k for k in sizes if sizes[k] > 0)
    if quota == 0 or not nonempty:
        return alloc
    total = sum(sizes.values())
    if quota >= total:
        return dict(sizes)
    if quota >= len(nonempty):
        for k in nonempty:
            alloc[k] = 1
    remaining = quota - sum(alloc.values())
    if remaining > 0:
        w = {k: float((weights or {}).get(k, 1.0)) * math.sqrt(sizes[k]) for k in nonempty}
        caps = {k: sizes[k] - alloc[k] for k in nonempty}
        for k, n in _largest_remainder(w, caps, remaining, rng).items():
            alloc[k] += n
    return alloc


def scale_quotas(quotas: dict[str, int], target: int, rng: random.Random) -> dict[str, int]:
    """Scale quotas proportionally so they sum to ``target`` (largest-remainder rounding)."""
    pos = {k: int(v) for k, v in quotas.items() if int(v) > 0}
    if not pos or target <= 0:
        return {k: 0 for k in quotas}
    scaled = _largest_remainder({k: float(v) for k, v in pos.items()}, {k: target for k in pos}, target, rng)
    return {k: scaled.get(k, 0) for k in quotas}


# ======================================================================================= field helpers
def _get(rec: VideoRecord, *keys: str, default: Any = None) -> Any:
    """First non-empty value of ``keys`` looked up in ``rec.strata`` then ``rec.labels``."""
    for src in (rec.strata or {}, rec.labels or {}):
        for k in keys:
            v = src.get(k)
            if v is not None and v != "":
                return v
    return default


def _text(rec: VideoRecord, *keys: str, default: str = UNKNOWN) -> str:
    v = _get(rec, *keys)
    return default if v is None else str(v)


def _num(rec: VideoRecord, *keys: str) -> Optional[float]:
    v = _get(rec, *keys)
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_TRUE = {"1", "true", "yes", "y", "t"}


def _flag(rec: VideoRecord, *keys: str, default: bool = False) -> bool:
    v = _get(rec, *keys)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in _TRUE


def _bins(cfg: Config) -> list[float]:
    return [float(b) for b in cfg.get("sampling.duration_bins_s", [300, 480, 720])]


def _dbin(rec: VideoRecord, bins: list[float]) -> str:
    """Duration bin from strata, else from duration_s, else the size bin, else 'unknown'."""
    b = _get(rec, "duration_bin")
    if b and str(b) != UNKNOWN:
        return str(b)
    b = duration_bin(rec.duration_s, bins)
    if b != UNKNOWN:
        return b
    sb = _get(rec, "size_bin")
    return f"size:{sb}" if sb else UNKNOWN


_SITE_RE = re.compile(r"S([12])")
_SITE_ID_RE = re.compile(r"_S([12])(?:_|$)")


def _site(rec: VideoRecord) -> str:
    """Normalise the Cataract-LMM site to 'S1'/'S2' (falls back to the video id suffix)."""
    s = _text(rec, "site", default="")
    low = s.lower()
    if "noor" in low:
        return "S2"
    if "farabi" in low:
        return "S1"
    m = _SITE_RE.search(s.upper()) or _SITE_ID_RE.search(rec.video_id.upper())
    return f"S{m.group(1)}" if m else (s or UNKNOWN)


def _sub(cfg: Config, name: str) -> dict:
    v = cfg.get(f"sampling.{name}", {})
    return v if isinstance(v, dict) else {}


# ======================================================================================= selection primitives
def pick(records: list[VideoRecord], k: int, rng: random.Random, priority: Optional[PriorityFn] = None) -> list[VideoRecord]:
    """``k`` records without replacement: shuffled, then stable-sorted by ``priority`` (lower first)
    so ties are random. Deterministic given ``rng``."""
    if k <= 0 or not records:
        return []
    recs = sorted(records, key=lambda r: r.sample_id)
    rng.shuffle(recs)
    if priority is not None:
        recs.sort(key=priority)
    return recs[:k]


def group_by(records: Iterable[VideoRecord], keyfn: StratumFn) -> dict[str, list[VideoRecord]]:
    out: dict[str, list[VideoRecord]] = defaultdict(list)
    for r in records:
        out[keyfn(r)].append(r)
    return {k: out[k] for k in sorted(out)}


def one_per_key(records: list[VideoRecord], keyfn: StratumFn, rng: random.Random,
                priority: Optional[PriorityFn] = None) -> tuple[list[VideoRecord], int]:
    """Keep one record per key (best ``priority`` first, random among ties). Returns (kept, n_dropped)."""
    kept: list[VideoRecord] = []
    for _key, grp in group_by(records, keyfn).items():
        kept.extend(pick(grp, 1, rng, priority))
    return kept, len(records) - len(kept)


@dataclass
class Pick:
    rec: VideoRecord
    stratum: str
    reason: str


@dataclass
class Group:
    """A sub-pool of one dataset with its own quota (e.g. cataract1k annotated / unannotated)."""
    name: str
    records: list[VideoRecord]
    quota: int
    stratum_of: StratumFn
    must_include: Optional[Callable[[VideoRecord], bool]] = None
    must_reason: str = "must-include rule"
    priority: Optional[PriorityFn] = None
    weights: Optional[dict[str, float]] = None
    describe: Optional[Describer] = None
    # filled by select_group
    allocation: dict[str, int] = field(default_factory=dict)
    pool_sizes: dict[str, int] = field(default_factory=dict)
    n_must: int = 0
    n_selected: int = 0


def _reason(ds: str, g: Group, r: VideoRecord, core: str) -> str:
    parts = [f"{ds}/{g.name}: {core}"]
    if g.weights:
        w = g.weights.get(g.stratum_of(r))
        if w is not None and w != 1.0:
            parts.append(f"category weight {w:g}")
    if g.describe is not None:
        try:
            extra = g.describe(r)
        except Exception:  # a describer must never break sampling
            extra = ""
        if extra:
            parts.append(extra)
    return "; ".join(parts)


def select_group(ds: str, g: Group, rng: random.Random, log=None) -> list[Pick]:
    """Must-includes first, then :func:`allocate` over the strata of the remaining records."""
    recs = sorted(g.records, key=lambda r: r.sample_id)
    quota = max(0, min(int(g.quota), len(recs)))
    g.pool_sizes = {k: len(v) for k, v in group_by(recs, g.stratum_of).items()}
    picks: list[Pick] = []
    chosen: set[str] = set()
    if g.must_include is not None and quota > 0:
        must = [r for r in recs if g.must_include(r)]
        if len(must) > quota and log is not None:
            log.warning("%s/%s: %d must-include records exceed the quota %d; a random subset is kept",
                        ds, g.name, len(must), quota)
        for r in pick(must, quota, rng, g.priority):
            picks.append(Pick(r, g.stratum_of(r), _reason(ds, g, r, g.must_reason)))
            chosen.add(r.sample_id)
        g.n_must = len(picks)
    rest = [r for r in recs if r.sample_id not in chosen]
    strata = group_by(rest, g.stratum_of)
    sizes = {k: len(v) for k, v in strata.items()}
    alloc = allocate(sizes, quota - len(picks), rng, g.weights)
    g.allocation = alloc
    for k in sorted(alloc):
        n = alloc[k]
        if n <= 0:
            continue
        for r in pick(strata[k], n, rng, g.priority):
            picks.append(Pick(r, k, _reason(ds, g, r, f"stratum '{k}': {n} of {sizes[k]} in pool")))
    g.n_selected = len(picks)
    return picks


@dataclass
class DatasetPlan:
    """Eligible pool + groups for one dataset (output of a planner)."""
    name: str
    quota: int
    inventory: list[VideoRecord]
    eligible: list[VideoRecord]
    groups: list[Group]
    stratum_of: StratumFn
    priority: Optional[PriorityFn] = None
    weights: Optional[dict[str, float]] = None
    describe: Optional[Describer] = None
    excluded: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ======================================================================================= planners (5.1)
def plan_cataract101(name: str, records: list[VideoRecord], quota: int, cfg: Config, rng: random.Random) -> DatasetPlan:
    bins = _bins(cfg)

    def stratum(r: VideoRecord) -> str:
        return f"{_text(r, 'experience')}|{_dbin(r, bins)}"

    def describe(r: VideoRecord) -> str:
        return f"surgeon {_text(r, 'surgeon', 'surgeon_id')}, {fmt_duration(r.duration_s)}"

    plan = DatasetPlan(name, quota, records, list(records), [], stratum, describe=describe)
    plan.groups = [Group("all", plan.eligible, quota, stratum, describe=describe)]
    if quota >= len(records):
        plan.notes.append(f"quota {quota} >= pool {len(records)}: every video is taken")
    else:
        plan.notes.append("balanced over experience x duration bin")
    return plan


def plan_cataract1k(name: str, records: list[VideoRecord], quota: int, cfg: Config, rng: random.Random) -> DatasetPlan:
    bins = _bins(cfg)
    c = _sub(cfg, "cataract1k")
    share = float(c.get("annotated_share", 0.87))
    rare_first = bool(c.get("rare_flag_first", True))

    def is_annot(r: VideoRecord) -> bool:
        return _flag(r, "annotated", default=bool(r.phase_segments))

    def flags_of(r: VideoRecord) -> list[str]:
        f = r.labels.get("flags") if r.labels else None
        return [str(x) for x in f] if isinstance(f, list) else []

    def is_rare(r: VideoRecord) -> bool:
        if _get(r, "rare_flag") is not None:
            return _flag(r, "rare_flag")
        return bool(flags_of(r)) or _flag(r, "has_suture") or _flag(r, "has_not_cataract")

    def stratum(r: VideoRecord) -> str:
        if is_annot(r):
            return f"annotated|{'rare' if is_rare(r) else 'common'}|{_dbin(r, bins)}"
        return f"unannotated|{_dbin(r, bins)}"

    def describe(r: VideoRecord) -> str:
        parts = [fmt_duration(r.duration_s)] if r.duration_s is not None else []
        fl = flags_of(r)
        if fl:
            parts.append("flags: " + ", ".join(fl))
        if _flag(r, "has_suture"):
            parts.append("suture")
        if _flag(r, "has_not_cataract"):
            parts.append("non-cataract segment")
        return ", ".join(parts)

    annotated = [r for r in records if is_annot(r)]
    unannot = [r for r in records if not is_annot(r)]
    q_ann = min(len(annotated), int(round(quota * share)))
    q_un = min(len(unannot), quota - q_ann)
    q_ann = min(len(annotated), quota - q_un)  # hand back to annotated when the unannotated pool is short
    plan = DatasetPlan(name, quota, records, annotated + unannot, [], stratum, describe=describe)
    plan.groups = [
        Group("annotated", annotated, q_ann, stratum,
              must_include=is_rare if rare_first else None,
              must_reason="rare-flag video (iris hooks / Malyugin ring / trypan blue / suture / non-cataract) included first",
              describe=describe),
        Group("unannotated", unannot, q_un, stratum, describe=describe),
    ]
    n_rare = sum(1 for r in annotated if is_rare(r))
    plan.notes.append(f"annotated share {share:.2f}: {q_ann} annotated + {q_un} unannotated of quota {quota} "
                      f"(pools {len(annotated)} / {len(unannot)}); {n_rare} rare-flag videos"
                      + (" included first" if rare_first else ""))
    return plan


def plan_lmm_phase(name: str, records: list[VideoRecord], quota: int, cfg: Config, rng: random.Random) -> DatasetPlan:
    bins = _bins(cfg)

    def idle_bin(r: VideoRecord) -> str:
        b = _get(r, "idle_share_bin")
        if b:
            return str(b)
        v = _num(r, "idle_share")
        if v is None:
            return UNKNOWN
        return "<0.2" if v < 0.2 else ("0.2-0.35" if v <= 0.35 else ">0.35")

    def stratum(r: VideoRecord) -> str:
        return f"{_site(r)}|{_dbin(r, bins)}|idle{idle_bin(r)}"

    def describe(r: VideoRecord) -> str:
        return f"{fmt_duration(r.duration_s)}, {_num(r, 'n_segments') or len(r.segments):g} segments"

    plan = DatasetPlan(name, quota, records, list(records), [], stratum, describe=describe)
    plan.groups = [Group("all", plan.eligible, quota, stratum, describe=describe)]
    plan.notes.append(f"quota {quota} >= pool {len(records)}: every video is taken" if quota >= len(records)
                      else "balanced over site x duration bin x idle-share bin")
    return plan


def _tertile_fn(records: list[VideoRecord]) -> StratumFn:
    """skill_tertile from strata when the adapter provides it, else computed from the averaged score."""
    if any(_get(r, "skill_tertile") is not None for r in records):
        return lambda r: _text(r, "skill_tertile")
    vals = sorted(v for v in (_num(r, "averaged", "Averaged") for r in records) if v is not None)
    if not vals:
        return lambda r: UNKNOWN
    lo, hi = vals[len(vals) // 3], vals[(2 * len(vals)) // 3]

    def f(r: VideoRecord) -> str:
        v = _num(r, "averaged", "Averaged")
        if v is None:
            return UNKNOWN
        return "low" if v < lo else ("mid" if v < hi else "high")
    return f


def plan_lmm_skill(name: str, records: list[VideoRecord], quota: int, cfg: Config, rng: random.Random) -> DatasetPlan:
    c = _sub(cfg, "lmm_skill")
    include_all = bool(c.get("include_all_adverse", True))
    tertile = _tertile_fn(records)

    def is_adverse(r: VideoRecord) -> bool:
        return _flag(r, "adverse_event", "adverse_events")

    def stratum(r: VideoRecord) -> str:
        return f"{tertile(r)}|{_site(r)}"

    def describe(r: VideoRecord) -> str:
        avg = _num(r, "averaged", "Averaged")
        s = f"averaged skill score {avg:g}" if avg is not None else ""
        if is_adverse(r):
            s += (", " if s else "") + "adverse event"
        return s

    plan = DatasetPlan(name, quota, records, list(records), [], stratum, describe=describe)
    plan.groups = [Group("all", plan.eligible, quota, stratum,
                         must_include=is_adverse if include_all else None,
                         must_reason="adverse-event clip (all adverse-event clips are included)",
                         describe=describe)]
    n_adv = sum(1 for r in records if is_adverse(r))
    plan.notes.append(f"{n_adv} adverse-event clips in pool" + (" (all included first)" if include_all else "")
                      + "; remainder balanced over skill tertile x site")
    return plan


def plan_lmm_raw(name: str, records: list[VideoRecord], quota: int, cfg: Config, rng: random.Random) -> DatasetPlan:
    bins = _bins(cfg)
    c = _sub(cfg, "lmm_raw")
    lo = float(c.get("min_duration_s", 300))
    hi = float(c.get("max_duration_s", 1500))
    s2_min = int(c.get("s2_min", 30))
    excl = bool(c.get("exclude_phase_subset", True))
    excluded: Counter = Counter()

    def in_range(r: VideoRecord) -> bool:
        return r.duration_s is not None and lo <= r.duration_s <= hi

    def stratum(r: VideoRecord) -> str:
        return f"{_site(r)}|{_dbin(r, bins)}"

    def priority(r: VideoRecord) -> int:
        return 0 if in_range(r) else 1

    def describe(r: VideoRecord) -> str:
        return fmt_duration(r.duration_s) + ("" if in_range(r) else f" (outside {lo:g}-{hi:g} s, S2 fallback)")

    base: list[VideoRecord] = []
    for r in records:
        if excl and _flag(r, "in_phase_subset"):
            excluded["in_phase_subset (raw video also used by lmm_phase)"] += 1
        else:
            base.append(r)
    s2_all = [r for r in base if _site(r) == "S2"]
    s1_all = [r for r in base if _site(r) != "S2"]
    s1 = [r for r in s1_all if in_range(r)]
    if len(s1_all) - len(s1):
        excluded[f"S1 duration outside [{lo:g}, {hi:g}] s or unknown"] += len(s1_all) - len(s1)
    s2_in = [r for r in s2_all if in_range(r)]
    notes: list[str] = []
    q_s2 = min(len(s2_all), quota, max(s2_min, int(round(quota * len(s2_in) / max(1, len(s2_in) + len(s1))))))
    if len(s2_in) >= q_s2:
        s2_pool = s2_in
        if len(s2_all) - len(s2_in):
            excluded[f"S2 duration outside [{lo:g}, {hi:g}] s (enough in-range S2 videos)"] += len(s2_all) - len(s2_in)
    else:
        s2_pool = s2_all
        notes.append(f"only {len(s2_in)} S2 videos within the duration range; all {len(s2_all)} S2 videos are "
                     f"eligible (in-range first) to reach s2_min={s2_min}")
    q_s1 = min(len(s1), quota - q_s2)
    notes.append(f"quota split: {q_s2} S2 (s2_min {s2_min}, {len(s2_pool)} eligible) + {q_s1} S1 ({len(s1)} eligible); "
                 f"balanced over duration bin within each site")
    plan = DatasetPlan(name, quota, records, s1 + s2_pool, [], stratum, priority=priority, describe=describe,
                       excluded=dict(excluded), notes=notes)
    plan.groups = [Group("S2", s2_pool, q_s2, stratum, priority=priority, describe=describe),
                   Group("S1", s1, q_s1, stratum, describe=describe)]
    return plan


def plan_migs(name: str, records: list[VideoRecord], quota: int, cfg: Config, rng: random.Random) -> DatasetPlan:
    c = _sub(cfg, "migs")
    mpo = int(c.get("min_per_operation_type", 1))

    def op(r: VideoRecord) -> str:
        return _text(r, "operation_type", "operation type")

    def stratum(r: VideoRecord) -> str:
        return f"{op(r)}|{_text(r, 'knife')}"

    def describe(r: VideoRecord) -> str:
        return f"{op(r)} with {_text(r, 'knife')} knife, {fmt_duration(r.duration_s)}"

    must_ids: set[str] = set()
    if mpo > 0:
        for _k, grp in group_by(records, op).items():
            must_ids.update(r.sample_id for r in pick(grp, mpo, rng))
    plan = DatasetPlan(name, quota, records, list(records), [], stratum, describe=describe)
    plan.groups = [Group("all", plan.eligible, quota, stratum,
                         must_include=(lambda r: r.sample_id in must_ids) if must_ids else None,
                         must_reason=f"guarantees >= {mpo} video(s) per operation type", describe=describe)]
    n_ops = len(group_by(records, op))
    plan.notes.append(f"{n_ops} operation types, each guaranteed >= {mpo}; remainder balanced over operation type x knife")
    return plan


def plan_ophnet(name: str, records: list[VideoRecord], quota: int, cfg: Config, rng: random.Random) -> DatasetPlan:
    c = _sub(cfg, "ophnet")
    min_ph = int(c.get("min_phases", 3))
    lo = float(c.get("min_case_duration_s", 60))
    hi = float(c.get("max_case_duration_s", 1200))
    unit = str(c.get("unit", "case"))
    max_per_case = int(c.get("max_per_case", 1))

    def n_phases(r: VideoRecord) -> float:
        v = _num(r, "n_phases")
        return v if v is not None else float(len({s.label for s in r.phase_segments}))

    def preferred(r: VideoRecord) -> bool:
        return n_phases(r) >= min_ph and r.duration_s is not None and lo <= r.duration_s <= hi

    def priority(r: VideoRecord) -> int:
        return 0 if preferred(r) else 1

    def stratum(r: VideoRecord) -> str:
        return _text(r, "primary_surgery", "primary_surgery_name", "surgery")

    def describe(r: VideoRecord) -> str:
        return f"{int(n_phases(r))} phases, {fmt_duration(r.duration_s)}" + ("" if preferred(r) else " (not preferred)")

    excluded: Counter = Counter()
    eligible = list(records)
    if unit == "clip" and max_per_case >= 1:
        def case_key(r: VideoRecord) -> str:
            return str(_get(r, "case_id", "case") or r.video_id.rsplit("_", 1)[0])
        before = len(eligible)
        kept: list[VideoRecord] = []
        for _k, grp in group_by(eligible, case_key).items():
            kept.extend(pick(grp, max_per_case, rng, priority))
        eligible = kept
        excluded[f"extra clips of the same case (max_per_case={max_per_case})"] += before - len(kept)
    plan = DatasetPlan(name, quota, records, eligible, [], stratum, priority=priority, describe=describe,
                       excluded=dict(excluded))
    plan.groups = [Group("all", eligible, quota, stratum, priority=priority, describe=describe)]
    n_pref = sum(1 for r in eligible if preferred(r))
    n_surg = len(group_by(eligible, stratum))
    plan.notes.append(f"{n_pref} of {len(eligible)} cases preferred (>= {min_ph} phases, {lo:g}-{hi:g} s) and filled "
                      f"first; sqrt-proportional allocation over {n_surg} primary surgeries, each >= 1 when the quota allows")
    return plan


def plan_ophora(name: str, records: list[VideoRecord], quota: int, cfg: Config, rng: random.Random) -> DatasetPlan:
    c = _sub(cfg, "ophora")
    one_per = bool(c.get("one_clip_per_source_video", True))
    min_words = int(c.get("min_instruction_words", 0) or 0)
    weights = {str(k): float(v) for k, v in (c.get("category_weights") or {}).items()}

    def words(r: VideoRecord) -> float:
        v = _num(r, "instruction_words")
        return v if v is not None else float(len(str((r.labels or {}).get("instruction", "")).split()))

    def src(r: VideoRecord) -> str:
        return str(_get(r, "source_video_id") or r.video_id.rsplit("_", 1)[0])

    def stratum(r: VideoRecord) -> str:
        return r.procedure_category or "other_mixed"

    def cat_weight(r: VideoRecord) -> float:
        return weights.get(stratum(r), 1.0)

    def clip_priority(r: VideoRecord) -> tuple:
        return (-cat_weight(r), -words(r))

    excluded: Counter = Counter()
    pool: list[VideoRecord] = []
    for r in records:
        if min_words and words(r) < min_words:
            excluded[f"instruction shorter than {min_words} words"] += 1
        else:
            pool.append(r)
    siblings = Counter(src(r) for r in pool)
    if one_per:
        pool, dropped = one_per_key(pool, src, rng, priority=clip_priority)
        if dropped:
            excluded["other clips of the same source video (one clip per source)"] += dropped

    def describe(r: VideoRecord) -> str:
        return (f"1 of {siblings[src(r)]} eligible clips of source video {src(r)} (highest category weight, "
                f"longest instruction); {int(words(r))}-word instruction")

    plan = DatasetPlan(name, quota, records, pool, [], stratum, weights=weights or None, describe=describe,
                       excluded=dict(excluded))
    plan.groups = [Group("all", pool, quota, stratum, weights=weights or None, describe=describe)]
    plan.notes.append(f"{len(pool)} eligible clips from {len(siblings)} source videos; allocation over procedure "
                      f"category with weights {json.dumps(weights) if weights else 'none'}")
    return plan


def plan_generic(name: str, records: list[VideoRecord], quota: int, cfg: Config, rng: random.Random) -> DatasetPlan:
    """Fallback: balance over the cross product of whatever strata the adapter provided."""
    keys = sorted({k for r in records for k in (r.strata or {})})

    def stratum(r: VideoRecord) -> str:
        return "|".join(f"{k}={_text(r, k)}" for k in keys) if keys else "all"

    plan = DatasetPlan(name, quota, records, list(records), [], stratum)
    plan.groups = [Group("all", plan.eligible, quota, stratum)]
    plan.notes.append(f"generic planner: balanced over strata {keys or ['(none)']}")
    return plan


PLANNERS: dict[str, Callable[..., DatasetPlan]] = {
    "cataract101": plan_cataract101,
    "cataract1k": plan_cataract1k,
    "lmm_phase": plan_lmm_phase,
    "lmm_skill": plan_lmm_skill,
    "lmm_raw": plan_lmm_raw,
    "migs": plan_migs,
    "ophnet": plan_ophnet,
    "ophora": plan_ophora,
}


# ======================================================================================= orchestration
@dataclass
class DatasetOutcome:
    plan: DatasetPlan
    picks: list[Pick]
    topups: int = 0


@dataclass
class SamplingResult:
    target: int
    seed: int
    quotas_configured: dict[str, int]
    quotas_effective: dict[str, int]
    outcomes: dict[str, DatasetOutcome]
    rounds: list[dict]
    missing_inventories: list[str]

    @property
    def total(self) -> int:
        return sum(len(o.picks) for o in self.outcomes.values())


def load_inventory(cfg: Config, name: str, log) -> Optional[list[VideoRecord]]:
    """VideoRecords of ``data/inventory/<name>.jsonl`` (None when the file does not exist)."""
    path = Path(cfg.paths.inventory) / f"{name}.jsonl"
    if not path.exists():
        return None
    recs: list[VideoRecord] = []
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(VideoRecord.from_dict(json.loads(line)))
            except Exception as e:
                bad += 1
                if bad <= 5:
                    log.warning("%s line %d unreadable (%s); skipped", path.name, ln, e)
    if bad:
        log.warning("%s: %d malformed lines skipped", path.name, bad)
    return recs


def effective_quotas(cfg: Config, target: int, rng: random.Random, log) -> tuple[dict[str, int], dict[str, int]]:
    """(configured, effective) quotas; effective ones are scaled to ``target`` when the sums differ."""
    raw = cfg.section("quotas") or {}
    configured = {str(k): int(v) for k, v in raw.items() if v is not None and int(v) > 0}
    if not configured:
        configured = scale_quotas({d: 1 for d in DATASETS}, target, rng)
        log.warning("no quotas configured; splitting the target evenly over %s", DATASETS)
    total = sum(configured.values())
    if total == target:
        return configured, dict(configured)
    effective = scale_quotas(configured, target, rng)
    log.info("quotas sum to %d but target is %d -> scaled proportionally: %s", total, target, effective)
    return configured, effective


def _dataset_order(quotas: dict[str, int]) -> list[str]:
    return [d for d in DATASETS if d in quotas] + sorted(d for d in quotas if d not in DATASETS)


def redistribute(outcomes: dict[str, DatasetOutcome], target: int, rng: random.Random, log, events: JsonlWriter,
                 max_rounds: int = 50) -> list[dict]:
    """DESIGN 5.3: move the deficit to datasets with remaining eligible pool, proportionally to that pool."""
    rounds: list[dict] = []
    for rnd in range(1, max_rounds + 1):
        selected = {ds: len(o.picks) for ds, o in outcomes.items()}
        deficit = target - sum(selected.values())
        if deficit <= 0:
            break
        short = {ds: o.plan.quota - selected[ds] for ds, o in outcomes.items() if o.plan.quota > selected[ds]}
        remaining = {ds: len(o.plan.eligible) - selected[ds] for ds, o in outcomes.items()
                     if len(o.plan.eligible) > selected[ds]}
        if not remaining:
            log.warning("shortfall of %d cannot be filled: every eligible pool is exhausted", deficit)
            rounds.append({"round": rnd, "deficit": deficit, "short": short, "remaining_pool": {}, "topups": {},
                           "note": "all eligible pools exhausted"})
            break
        extra = {k: v for k, v in _largest_remainder({k: float(v) for k, v in remaining.items()},
                                                     remaining, deficit, rng).items() if v > 0}
        short_txt = ", ".join(f"{k} -{v}" for k, v in sorted(short.items())) or "none"
        for ds in sorted(extra):
            o = outcomes[ds]
            chosen = {p.rec.sample_id for p in o.picks}
            rest = [r for r in o.plan.eligible if r.sample_id not in chosen]
            g = Group(f"topup-r{rnd}", rest, extra[ds], o.plan.stratum_of, priority=o.plan.priority,
                      weights=o.plan.weights, describe=o.plan.describe)
            new = select_group(ds, g, rng, log)
            for p in new:
                p.reason = f"shortfall redistribution round {rnd} (+{extra[ds]} to {ds}; short: {short_txt}); {p.reason}"
            o.picks.extend(new)
            o.topups += len(new)
            o.plan.groups.append(g)
        rounds.append({"round": rnd, "deficit": deficit, "short": short, "remaining_pool": remaining, "topups": extra})
        log.info("redistribution round %d: deficit %d (short: %s) -> top-ups %s", rnd, deficit, short_txt, extra)
        events.write({"event": "redistribution", "round": rnd, "deficit": deficit, "short": short, "topups": extra})
    return rounds


def run_sampling(cfg: Config, target: int, seed: int, log, events: JsonlWriter) -> SamplingResult:
    rng = random.Random(seed)
    configured, quotas = effective_quotas(cfg, target, rng, log)
    outcomes: dict[str, DatasetOutcome] = {}
    missing: list[str] = []
    for ds in _dataset_order(quotas):
        q = quotas[ds]
        recs = load_inventory(cfg, ds, log)
        if recs is None:
            log.warning("no inventory for %s (%s); its quota of %d is redistributed", ds,
                        Path(cfg.paths.inventory) / f"{ds}.jsonl", q)
            missing.append(ds)
            recs = []
        planner = PLANNERS.get(ds, plan_generic)
        try:
            plan = planner(ds, recs, q, cfg, rng)
        except Exception as e:
            log.error("%s planner failed (%s: %s); falling back to the generic strata planner", ds,
                      type(e).__name__, e, exc_info=True)
            plan = plan_generic(ds, recs, q, cfg, rng)
        picks: list[Pick] = []
        for g in plan.groups:
            picks.extend(select_group(ds, g, rng, log))
        outcomes[ds] = DatasetOutcome(plan, picks)
        short = q - len(picks)
        log.info("%s: inventory %d | eligible %d | quota %d | selected %d%s", ds, len(recs), len(plan.eligible), q,
                 len(picks), f" | SHORT by {short}" if short > 0 else "")
        for note in plan.notes:
            log.info("  %s: %s", ds, note)
        for k, v in sorted(plan.excluded.items()):
            log.info("  %s: excluded %d (%s)", ds, v, k)
        for g in plan.groups:
            log.debug("  %s/%s: quota %d, must-include %d, allocation %s", ds, g.name, g.quota, g.n_must, g.allocation)
        events.write({"event": "dataset_selected", "dataset": ds, "inventory": len(recs), "eligible": len(plan.eligible),
                      "quota": q, "selected": len(picks), "excluded": plan.excluded,
                      "groups": [{"name": g.name, "quota": g.quota, "must": g.n_must, "allocation": g.allocation}
                                 for g in plan.groups]})
    rounds = redistribute(outcomes, target, rng, log, events)
    return SamplingResult(target, seed, configured, quotas, outcomes, rounds, missing)


# ======================================================================================= outputs (5.4)
def build_manifest(result: SamplingResult, log=None) -> list[SampleRecord]:
    """SampleRecords in the canonical benchmark order (DESIGN.md 13.1: proportional interleave across
    datasets, within a dataset sorted by sample_id); ``sample_index`` = 0-based rank in that order, so
    annotation batch K of size N is rows ``[(K-1)*N, K*N)`` of the manifest."""
    rows: list[Pick] = []
    seen: set[str] = set()
    for ds in result.outcomes:
        for p in sorted(result.outcomes[ds].picks, key=lambda p: p.rec.sample_id):
            if p.rec.sample_id in seen:  # cannot happen by construction; keep the manifest clean anyway
                if log is not None:
                    log.error("duplicate selection of %s dropped", p.rec.sample_id)
                continue
            seen.add(p.rec.sample_id)
            rows.append(p)
    ordered = order_samples([SampleRecord.from_video(p.rec, p.stratum, p.reason, -1) for p in rows])
    for i, m in enumerate(ordered):
        m.sample_index = i
    return ordered


def manifest_csv_row(m: SampleRecord) -> dict:
    return {"sample_id": m.sample_id, "dataset": m.dataset, "video_id": m.video_id, "procedure": m.procedure,
            "procedure_category": m.procedure_category,
            "duration_s": "" if m.duration_s is None else round(float(m.duration_s), 3),
            "stratum_key": m.stratum_key, "selection_reason": m.selection_reason, "source_kind": m.source_kind,
            "first_source_path": m.source_paths[0] if m.source_paths else ""}


def build_report(result: SamplingResult, cfg: Config, args_snapshot: dict) -> dict:
    datasets: dict[str, dict] = {}
    for ds, o in result.outcomes.items():
        plan = o.plan
        pool_by = Counter(plan.stratum_of(r) for r in plan.eligible)
        sel_by = Counter(p.stratum for p in o.picks)
        strata = [{"stratum": k, "pool": pool_by.get(k, 0), "selected": sel_by.get(k, 0)}
                  for k in sorted(set(pool_by) | set(sel_by))]
        groups = [{"name": g.name, "pool": len(g.records), "quota": g.quota, "must_include": g.n_must,
                   "selected": g.n_selected, "allocation": g.allocation, "pool_by_stratum": g.pool_sizes}
                  for g in plan.groups]
        initial = len(o.picks) - o.topups
        datasets[ds] = {
            "inventory": len(plan.inventory), "eligible": len(plan.eligible),
            "quota_configured": result.quotas_configured.get(ds), "quota": plan.quota,
            "selected": len(o.picks), "selected_initial": initial, "shortfall": max(0, plan.quota - initial),
            "topup": o.topups, "inventory_missing": ds in result.missing_inventories,
            "excluded": dict(plan.excluded), "notes": list(plan.notes), "groups": groups, "strata": strata,
            "procedure_categories": dict(Counter(p.rec.procedure_category for p in o.picks)),
        }
    return {
        "generated_at": now_iso(), "seed": result.seed, "target": result.target, "total_selected": result.total,
        "quotas_configured": result.quotas_configured, "quotas_effective": result.quotas_effective,
        "datasets": datasets, "redistribution_log": result.rounds, "missing_inventories": result.missing_inventories,
        "config": {"seed": cfg.seed, "target_total": cfg.get("target_total"), "quotas": cfg.section("quotas"),
                   "sampling": cfg.section("sampling"), "inventory_dir": str(cfg.paths.inventory)},
        "args": args_snapshot,
    }


def render_report_md(report: dict) -> str:
    L = ["# Sampling report", "",
         f"- generated: {report['generated_at']}",
         f"- seed: {report['seed']}",
         f"- target: {report['target']}; selected: **{report['total_selected']}**"]
    if report.get("missing_inventories"):
        L.append(f"- missing inventories (quota redistributed): {', '.join(report['missing_inventories'])}")
    L += ["", "## Overview", ""]
    rows = []
    for ds, d in report["datasets"].items():
        rows.append([ds, d["inventory"], d["eligible"], d["quota_configured"], d["quota"], d["selected"],
                     d["shortfall"], d["topup"]])
    rows.append(["**total**", sum(d["inventory"] for d in report["datasets"].values()),
                 sum(d["eligible"] for d in report["datasets"].values()),
                 sum(report["quotas_configured"].values()), sum(report["quotas_effective"].values()),
                 report["total_selected"], sum(d["shortfall"] for d in report["datasets"].values()),
                 sum(d["topup"] for d in report["datasets"].values())])
    L.append(md_table(["dataset", "inventory", "eligible", "quota (config)", "quota (effective)", "selected",
                       "shortfall", "top-up"], rows))
    L += ["", "## Shortfall redistribution", ""]
    if not report["redistribution_log"]:
        L.append("No shortfall: every dataset filled its quota from its own eligible pool.")
    else:
        rrows = []
        for r in report["redistribution_log"]:
            rrows.append([r["round"], r["deficit"],
                          ", ".join(f"{k} -{v}" for k, v in sorted(r.get("short", {}).items())) or "-",
                          ", ".join(f"{k} +{v}" for k, v in sorted(r.get("topups", {}).items())) or r.get("note", "-")])
        L.append(md_table(["round", "deficit", "short datasets", "top-ups"], rrows))
    for ds, d in report["datasets"].items():
        L += ["", f"## {ds}", "",
              f"inventory {d['inventory']} -> eligible {d['eligible']} -> quota {d['quota']} -> selected {d['selected']}"
              + (f" (initial shortfall {d['shortfall']}, top-up {d['topup']})" if d["shortfall"] or d["topup"] else "")]
        if d.get("inventory_missing"):
            L.append("**inventory file missing**")
        for k, v in d["excluded"].items():
            L.append(f"- excluded {v}: {k}")
        for n in d["notes"]:
            L.append(f"- {n}")
        if d["groups"]:
            L += ["", "Groups:", "", md_table(["group", "pool", "quota", "must-include", "selected"],
                                              [[g["name"], g["pool"], g["quota"], g["must_include"], g["selected"]]
                                               for g in d["groups"]])]
        if d["strata"]:
            tot = max(1, d["selected"])
            L += ["", "Strata (pool -> selected):", "",
                  md_table(["stratum", "pool", "selected", "share of selected"],
                           [[s["stratum"], s["pool"], s["selected"], f"{100.0 * s['selected'] / tot:.0f}%"]
                            for s in d["strata"]])]
        if d.get("procedure_categories"):
            L.append("")
            L.append("Procedure categories of the selection: "
                     + ", ".join(f"{k}: {v}" for k, v in sorted(d["procedure_categories"].items())))
    return "\n".join(L) + "\n"


def write_outputs(cfg: Config, manifest: list[SampleRecord], report: dict, log) -> dict[str, str]:
    sd = Path(cfg.paths.sample)
    paths = {
        "manifest_jsonl": sd / "sample_manifest.jsonl",
        "manifest_csv": sd / "sample_manifest.csv",
        "report_json": sd / "sampling_report.json",
        "report_md": sd / "sampling_report.md",
    }
    write_jsonl(paths["manifest_jsonl"], (m.to_dict() for m in manifest))
    write_csv(paths["manifest_csv"], [manifest_csv_row(m) for m in manifest], MANIFEST_CSV_FIELDS)
    write_json(paths["report_json"], report)
    paths["report_md"].write_text(render_report_md(report), encoding="utf-8")
    for k, p in paths.items():
        log.info("wrote %s -> %s", k, p)
    return {k: str(v) for k, v in paths.items()}


# ======================================================================================= CLI
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target", type=int, default=None, metavar="N",
                   help="total number of videos to select (default: config target_total; quotas are scaled)")
    p.add_argument("--seed", type=int, default=None, metavar="S", help="random seed (default: config seed)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print the allocation report without writing the manifest")


def main(args: argparse.Namespace, cfg: Config) -> int:
    cfg.ensure_dirs()
    log = get_logger(STAGE, cfg)
    started = now_iso()
    t0 = time.time()
    seed = cfg.seed if getattr(args, "seed", None) is None else int(args.seed)
    target = int(cfg.get("target_total", 1500)) if getattr(args, "target", None) is None else int(args.target)
    dry = bool(getattr(args, "dry_run", False))
    if target <= 0:
        log.error("target must be positive (got %d)", target)
        return 1
    log.info("sample start: target=%d seed=%d dry_run=%s inventory=%s", target, seed, dry, cfg.paths.inventory)
    log.info("config snapshot: quotas=%s sampling=%s", json.dumps(cfg.section("quotas")),
             json.dumps(cfg.section("sampling"), default=str))
    events = JsonlWriter(Path(cfg.paths.logs) / "sample_events.jsonl")
    events.write({"event": "start", "stage": STAGE, "target": target, "seed": seed, "dry_run": dry})

    try:
        result = run_sampling(cfg, target, seed, log, events)
    except Exception:
        log.exception("sampling failed")
        return 1
    args_snapshot = {k: v for k, v in vars(args).items() if not k.startswith("_") and k != "func"} \
        if hasattr(args, "__dict__") else {}
    report = build_report(result, cfg, args_snapshot)
    md = render_report_md(report)
    summary = {"target": target, "seed": seed, "selected": result.total,
               "per_dataset": {ds: len(o.picks) for ds, o in result.outcomes.items()},
               "redistribution_rounds": len(result.rounds), "missing_inventories": result.missing_inventories,
               "dry_run": dry, "elapsed_s": round(time.time() - t0, 1)}
    if result.total == 0:
        log.error("nothing selected: no inventories found under %s (run `./ophbench inventory` first)",
                  cfg.paths.inventory)
        record_run(cfg, STAGE, args, summary, started_at=started)
        return 1
    if dry:
        print(md)
        log.info("dry run: selected %d of target %d; manifest NOT written", result.total, target)
    else:
        manifest = build_manifest(result, log)
        write_outputs(cfg, manifest, report, log)
        log.info("manifest: %d samples (%s)", len(manifest),
                 ", ".join(f"{ds}={n}" for ds, n in summary["per_dataset"].items()))
    if result.total < target:
        log.warning("selected %d < target %d: eligible pools exhausted", result.total, target)
    events.write({"event": "end", "stage": STAGE, **summary})
    record_run(cfg, STAGE, args, summary, started_at=started)
    return 0
