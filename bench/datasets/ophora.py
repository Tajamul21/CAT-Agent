"""Ophora-160K adapter (DESIGN.md §3.6): the 28K filtered subset of ~5.5 s narrated YouTube clips.

Inputs under ``<datasets_root>/Ophora-160K``: ``ophora28k.csv`` (``clip id,instruction``) and
``clips/<clip id>.mp4`` (~162k files; listed with a single ``os.scandir`` that is cached on the
adapter). Rows whose clip file is missing are skipped with a WARNING.

Procedure category and label come from the ordered first-match keyword rules in
``resources/procedure_taxonomy.yaml: ophora_keyword_rules``. Matching runs on the lower-cased
instruction with punctuation replaced by spaces and padded with a space on both sides, so
word-boundary rules such as ``" iol"`` or ``" dcr"`` behave as intended. A rule with an empty
``any_of`` list is a catch-all fallback.
"""
from __future__ import annotations

import csv
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from bench.datasets.base import DatasetAdapter
from bench.schema import PROCEDURE_CATEGORIES, VideoRecord
from bench.util import read_csv_rows

OPHORA_DIR = "Ophora-160K"
SUBSET_CSV = "ophora28k.csv"
CLIPS_SUBDIR = "clips"
TAXONOMY_YAML = "procedure_taxonomy.yaml"
NOMINAL_DURATION_S = 5.5
FALLBACK_CATEGORY = "other_mixed"
FALLBACK_LABEL = "Other ophthalmic surgery"
DURATION_NOTE = f"duration unknown until probed (Ophora clips average ~{NOMINAL_DURATION_S} s)"
LEN_BINS: tuple[tuple[int, str], ...] = ((12, "<12"), (21, "12-20"))  # else ">20"

_NON_WORD_RE = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class KeywordRule:
    category: str
    label: str
    terms: tuple[str, ...]  # lower-cased; empty => catch-all


# ------------------------------------------------------------------------------------ pure helpers
def load_keyword_rules(path: Path, log: Any) -> list[KeywordRule]:
    """Ordered rules from the taxonomy YAML ([] after a WARNING when unreadable)."""
    try:
        with open(path, encoding="utf-8") as fh:
            tax = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as e:
        log.warning("ophora: cannot read taxonomy %s: %s (everything -> %s)", path, e, FALLBACK_CATEGORY)
        return []
    rules: list[KeywordRule] = []
    for raw in tax.get("ophora_keyword_rules") or []:
        category = str(raw.get("category") or FALLBACK_CATEGORY)
        if category not in PROCEDURE_CATEGORIES:
            log.warning("ophora: taxonomy category %r is not a known procedure category", category)
        terms = tuple(str(t).lower() for t in (raw.get("any_of") or []) if str(t).strip())
        rules.append(KeywordRule(category=category, label=str(raw.get("label") or category), terms=terms))
    return rules


def normalise_instruction(text: str) -> str:
    """Lower-case, punctuation -> space, collapse whitespace, pad with one space on each side."""
    t = _NON_WORD_RE.sub(" ", (text or "").lower())
    return " " + " ".join(t.split()) + " "


def classify_instruction(text: str, rules: list[KeywordRule]) -> tuple[str, str, list[str]]:
    """First matching rule -> (category, label, matched terms). Empty-term rules match everything."""
    hay = normalise_instruction(text)
    for rule in rules:
        if not rule.terms:
            return rule.category, rule.label, []
        matched = [t for t in rule.terms if t in hay]
        if matched:
            return rule.category, rule.label, matched
    return FALLBACK_CATEGORY, FALLBACK_LABEL, []


def instruction_len_bin(n_words: int) -> str:
    for upper, label in LEN_BINS:
        if n_words < upper:
            return label
    return ">20"


def split_clip_id(clip_id: str) -> tuple[str, Optional[int]]:
    """'X3jTUYMflCk_26' -> ('X3jTUYMflCk', 26); ids without a numeric suffix keep index None."""
    if "_" in clip_id:
        src, idx = clip_id.rsplit("_", 1)
        if idx.isdigit():
            return src, int(idx)
    return clip_id, None


# ------------------------------------------------------------------------------------ adapter
class OphoraAdapter(DatasetAdapter):
    """26,592 Ophora-28K clips with instruction captions (one record per clip file present)."""

    name = "ophora"

    def __init__(self, cfg: Any, log: Any = None):
        super().__init__(cfg, log)
        self.rules = load_keyword_rules(Path(cfg.paths.resources) / TAXONOMY_YAML, self.log)
        self._clip_names: Optional[set[str]] = None

    @property
    def dataset_root(self) -> Path:
        return self.root / OPHORA_DIR

    @property
    def clips_dir(self) -> Path:
        return self.dataset_root / CLIPS_SUBDIR

    def clip_names(self) -> set[str]:
        """File names under ``clips/`` from a single cached ``os.scandir``."""
        if self._clip_names is None:
            names: set[str] = set()
            if self.clips_dir.is_dir():
                with os.scandir(self.clips_dir) as it:
                    names = {e.name for e in it}
            else:
                self.log.warning("ophora: clips folder missing: %s", self.clips_dir)
            self._clip_names = names
            self.log.info("ophora: %d files under %s", len(names), self.clips_dir)
        return self._clip_names

    def classify(self, instruction: str) -> tuple[str, str, list[str]]:
        return classify_instruction(instruction, self.rules)

    def iter_records(self) -> Iterator[VideoRecord]:
        csv_path = self.dataset_root / SUBSET_CSV
        if not csv_path.exists():
            self.log.warning("ophora: subset table missing: %s", csv_path)
            return
        try:
            rows = read_csv_rows(csv_path)
        except (OSError, csv.Error) as e:
            self.log.warning("ophora: cannot read %s: %s", csv_path, e)
            return
        names = self.clip_names()
        seen: set[str] = set()
        missing = 0
        categories: Counter[str] = Counter()
        for row in rows:
            clip_id = (row.get("clip id") or row.get("clip_id") or "").strip()
            if not clip_id or clip_id in seen:
                continue
            seen.add(clip_id)
            fname = f"{clip_id}.mp4"
            if fname not in names:
                missing += 1
                self.log.warning("ophora: clip %s listed in %s has no file under clips/; skipped", clip_id, SUBSET_CSV)
                continue
            rec = self._record(clip_id, fname, (row.get("instruction") or "").strip())
            categories[rec.procedure_category] += 1
            yield rec
        self.log.info("ophora: %d records (%d listed clips missing); categories: %s",
                      sum(categories.values()), missing, dict(categories.most_common()))

    def _record(self, clip_id: str, fname: str, instruction: str) -> VideoRecord:
        category, label, matched = self.classify(instruction)
        source_video_id, clip_index = split_clip_id(clip_id)
        n_words = len(instruction.split())
        labels: dict[str, Any] = {
            "instruction": instruction,
            "source_video_id": source_video_id,
            "clip_index": clip_index,
            "in_filtered_28k": True,
            "instruction_words": n_words,
            "category_keywords": matched,
            "nominal_duration_s": NOMINAL_DURATION_S,
        }
        strata = {"procedure_category": category, "instruction_len_bin": instruction_len_bin(n_words)}
        return self.new_record(
            clip_id, source_kind="file", source_paths=[str(self.clips_dir / fname)], duration_s=None,
            procedure=label, procedure_category=category, labels=labels, strata=strata, notes=[DURATION_NOTE],
        )
