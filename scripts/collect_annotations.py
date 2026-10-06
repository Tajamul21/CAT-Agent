#!/usr/bin/env python3
"""Merge clinician annotations and summarise them (DESIGN.md §9).

Inputs
  <data_dir>/annotations/*.json   ANNOTATION RECORDs {ui_version, annotator, exported_at, samples}
                                  written by scripts/serve_ui.py (Submit) or exported from the UI
  --sheet-csv questions.csv       CSV export(s) of the Apps Script "questions" sheet (repeatable)
  --samples-csv samples.csv       CSV export(s) of the Apps Script "samples" sheet (repeatable)
  ui/data/samples/<id>.json | data/qa/<id>.json     question categories (optional, for statistics)
  ui/data/index.json | data/sample/sample_manifest.jsonl   dataset / procedure category (optional)

Outputs (default <data_dir>/annotations/)
  merged.jsonl     one row per annotator x sample x question (latest version of each)
  per_sample.csv   per-sample selected-question counts, consensus, flags
  summary.json     all statistics, machine readable
  summary.md       human-readable report (selection rates, correctness, ratings, agreement)

Usage
  python3 scripts/collect_annotations.py [--data-dir DIR] [--sheet-csv F ...] [--samples-csv F ...]
                                         [--only-done] [--out-dir DIR] [--ui-dir DIR] [-v]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.config import Config, load_config  # noqa: E402
from bench.log import get_logger, now_iso, record_run  # noqa: E402
from bench.util import iter_jsonl, md_table, read_csv_rows, write_json, write_jsonl  # noqa: E402

STAGE = "collect_annotations"
RATING_KEYS = ("relevance", "difficulty", "agentic", "clarity")
CORRECTNESS_VALUES = ("correct", "partial", "incorrect", "cannot_verify", "")
Q_FIELDS = ("selected", "correctness", "relevance", "difficulty", "agentic", "clarity", "edited",
            "question_edit", "answer_edit", "comment")


# ====================================================================== data model
@dataclass
class QuestionAnn:
    selected: bool = False
    correctness: str = ""
    relevance: Optional[int] = None
    difficulty: Optional[int] = None
    agentic: Optional[int] = None
    clarity: Optional[int] = None
    edited: bool = False
    question_edit: str = ""
    answer_edit: str = ""
    comment: str = ""

    @classmethod
    def from_any(cls, d: Any) -> "QuestionAnn":
        d = d if isinstance(d, dict) else {}
        q = cls(
            selected=to_bool(d.get("selected")),
            correctness=str(d.get("correctness") or ""),
            relevance=to_rating(d.get("relevance")),
            difficulty=to_rating(d.get("difficulty")),
            agentic=to_rating(d.get("agentic")),
            clarity=to_rating(d.get("clarity")),
            edited=to_bool(d.get("edited")),
            question_edit=str(d.get("question_edit") or ""),
            answer_edit=str(d.get("answer_edit") or ""),
            comment=str(d.get("comment") or ""),
        )
        if q.correctness not in CORRECTNESS_VALUES:
            q.correctness = ""
        q.edited = q.edited or bool(q.question_edit) or bool(q.answer_edit)
        return q


@dataclass
class SampleAnn:
    annotator: str
    sample_id: str
    status: str = "in_progress"
    updated_at: str = ""
    received_at: str = ""
    flags: list[str] = field(default_factory=list)
    comment: str = ""
    questions: dict[str, QuestionAnn] = field(default_factory=dict)
    source: str = ""

    @property
    def version_key(self) -> tuple[str, str]:
        return (self.updated_at or "", self.received_at or "")

    @property
    def n_selected(self) -> int:
        return sum(1 for q in self.questions.values() if q.selected)


# ====================================================================== parsing helpers
def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y", "x")


def to_rating(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 5 else None


def split_flags(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if x]
    s = str(v or "").strip()
    if not s:
        return []
    for sep in (";", ",", "|"):
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]
    return [s]


def dataset_of(sample_id: str) -> str:
    return sample_id.split("__", 1)[0] if "__" in sample_id else ""


def mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def pct(n: int, d: int) -> Optional[float]:
    return round(100.0 * n / d, 1) if d else None


# ====================================================================== loading
def iter_records(obj: Any) -> Iterable[dict]:
    """Yield ANNOTATION RECORD dicts from a parsed JSON file (single record or list of records)."""
    if isinstance(obj, dict) and isinstance(obj.get("samples"), dict):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from iter_records(x)


def load_json_records(paths: list[Path], log: logging.Logger) -> list[SampleAnn]:
    out: list[SampleAnn] = []
    for p in paths:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", p.name, exc)
            continue
        n_before = len(out)
        for rec in iter_records(obj):
            annotator = str(rec.get("annotator") or "anonymous")
            received = str(rec.get("received_at") or rec.get("exported_at") or "")
            for sid, s in rec["samples"].items():
                if not isinstance(s, dict):
                    continue
                sa = SampleAnn(
                    annotator=annotator, sample_id=str(sid),
                    status="done" if s.get("status") == "done" else "in_progress",
                    updated_at=str(s.get("updated_at") or ""), received_at=received,
                    flags=split_flags(s.get("flags")), comment=str(s.get("comment") or ""),
                    source=p.name,
                )
                qs = s.get("questions") if isinstance(s.get("questions"), dict) else {}
                for qid, q in qs.items():
                    sa.questions[str(qid)] = QuestionAnn.from_any(q)
                out.append(sa)
        if len(out) == n_before:
            log.warning("%s: no annotation records found", p.name)
        else:
            log.info("%s: %d sample annotations", p.name, len(out) - n_before)
    return out


def activity_stats(paths: list[Path], log: logging.Logger) -> dict:
    """Per clinician: work done and active time, from the latest export/progress file of each annotator."""
    latest: dict[str, tuple[str, dict]] = {}
    for p in paths:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - already reported by load_json_records
            continue
        for rec in iter_records(obj):
            who = str(rec.get("annotator") or "anonymous")
            ts = str(rec.get("exported_at") or rec.get("received_at") or "")
            if who not in latest or ts >= latest[who][0]:
                latest[who] = (ts, rec)
    out: dict[str, dict] = {}
    for who, (ts, rec) in sorted(latest.items()):
        samples = {k: v for k, v in (rec.get("samples") or {}).items() if isinstance(v, dict)}
        act = rec.get("activity") if isinstance(rec.get("activity"), dict) else {}
        by_sample = act.get("by_sample") if isinstance(act.get("by_sample"), dict) else {}
        def secs(sid: str, s: dict) -> float:
            return float(s.get("time_spent_s") or by_sample.get(sid) or 0)
        done = {k: v for k, v in samples.items() if v.get("status") == "done"}
        qs = [q for s in samples.values() for q in (s.get("questions") or {}).values() if isinstance(q, dict)]
        sessions = [s for s in (act.get("sessions") or []) if isinstance(s, dict)]
        active = float(act.get("total_active_s") or sum(float(x or 0) for x in by_sample.values()) or
                       sum(secs(k, v) for k, v in samples.items()))
        done_times = [secs(k, v) for k, v in done.items() if secs(k, v) > 0]
        out[who] = {
            "file_exported_at": ts, "videos_started": len(samples), "videos_done": len(done),
            "questions_selected": sum(1 for q in qs if q.get("selected")),
            "questions_edited": sum(1 for q in qs if q.get("edited")),
            "active_s": round(active), "active_h": round(active / 3600, 2),
            "avg_min_per_done_video": round(sum(done_times) / len(done_times) / 60, 1) if done_times else None,
            "sessions": int(act.get("n_sessions") or len(sessions)),
            "first_activity": min((str(s.get("started_at")) for s in sessions), default=None),
            "last_activity": max((str(s.get("last_at") or s.get("started_at")) for s in sessions), default=None),
        }
    return out


def load_sheet_csvs(question_csvs: list[Path], sample_csvs: list[Path], log: logging.Logger) -> list[SampleAnn]:
    """Rebuild SampleAnn objects from Apps Script sheet exports (latest row per key wins)."""
    q_latest: dict[tuple[str, str, str], dict] = {}
    for p in question_csvs:
        try:
            rows = read_csv_rows(p)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", p, exc)
            continue
        for r in rows:
            key = (str(r.get("annotator") or ""), str(r.get("sample_id") or ""), str(r.get("qid") or ""))
            if not key[1] or not key[2]:
                continue
            vk = (str(r.get("updated_at") or ""), str(r.get("received_at") or ""))
            if key not in q_latest or vk >= (str(q_latest[key].get("updated_at") or ""), str(q_latest[key].get("received_at") or "")):
                q_latest[key] = r
        log.info("%s: %d question rows", p.name, len(rows))
    s_latest: dict[tuple[str, str], dict] = {}
    for p in sample_csvs:
        try:
            rows = read_csv_rows(p)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", p, exc)
            continue
        for r in rows:
            key = (str(r.get("annotator") or ""), str(r.get("sample_id") or ""))
            if not key[1]:
                continue
            vk = (str(r.get("updated_at") or ""), str(r.get("received_at") or ""))
            if key not in s_latest or vk >= (str(s_latest[key].get("updated_at") or ""), str(s_latest[key].get("received_at") or "")):
                s_latest[key] = r
        log.info("%s: %d sample rows", p.name, len(rows))

    by_sample: dict[tuple[str, str], SampleAnn] = {}
    for (annotator, sid, qid), r in q_latest.items():
        sa = by_sample.get((annotator, sid))
        if sa is None:
            sa = SampleAnn(annotator=annotator, sample_id=sid, source="sheet")
            by_sample[(annotator, sid)] = sa
        sa.questions[qid] = QuestionAnn.from_any(r)
        sa.updated_at = max(sa.updated_at, str(r.get("updated_at") or ""))
        sa.received_at = max(sa.received_at, str(r.get("received_at") or ""))
    for (annotator, sid), r in s_latest.items():
        sa = by_sample.get((annotator, sid))
        if sa is None:
            sa = SampleAnn(annotator=annotator, sample_id=sid, source="sheet")
            by_sample[(annotator, sid)] = sa
        sa.status = "done" if str(r.get("status") or "") == "done" else sa.status
        sa.flags = split_flags(r.get("flags"))
        sa.comment = str(r.get("comment") or "")
        sa.updated_at = max(sa.updated_at, str(r.get("updated_at") or ""))
        sa.received_at = max(sa.received_at, str(r.get("received_at") or ""))
    return list(by_sample.values())


def dedupe_latest(items: list[SampleAnn]) -> dict[tuple[str, str], SampleAnn]:
    """Keep the newest (updated_at, received_at) version per (annotator, sample_id)."""
    best: dict[tuple[str, str], SampleAnn] = {}
    for sa in items:
        key = (sa.annotator, sa.sample_id)
        if key not in best or sa.version_key >= best[key].version_key:
            best[key] = sa
    return best


# ====================================================================== sample metadata lookups
class SampleInfo:
    """Lazy lookup of dataset / category / question categories for sample ids."""

    def __init__(self, cfg: Config, ui_dir: Path, log: logging.Logger):
        self.cfg = cfg
        self.ui_dir = ui_dir
        self.log = log
        self._index: dict[str, dict] = {}
        self._qcache: dict[str, dict] = {}
        self._load_index()

    def _load_index(self) -> None:
        idx = self.ui_dir / "data" / "index.json"
        if idx.is_file():
            try:
                for r in json.loads(idx.read_text(encoding="utf-8")):
                    self._index[r["sample_id"]] = r
                return
            except Exception as exc:  # noqa: BLE001
                self.log.warning("could not read %s: %s", idx, exc)
        manifest = Path(self.cfg.paths.sample) / "sample_manifest.jsonl"
        if manifest.is_file():
            for r in iter_jsonl(manifest):
                self._index[r["sample_id"]] = r

    def dataset(self, sid: str) -> str:
        r = self._index.get(sid)
        return str(r.get("dataset") or dataset_of(sid)) if r else dataset_of(sid)

    def category(self, sid: str) -> str:
        r = self._index.get(sid)
        if r and r.get("procedure_category"):
            return str(r["procedure_category"])
        return str(self._questions(sid).get("_procedure_category") or "")

    def _questions(self, sid: str) -> dict:
        if sid in self._qcache:
            return self._qcache[sid]
        info: dict[str, Any] = {}
        for p in (self.ui_dir / "data" / "samples" / f"{sid}.json", Path(self.cfg.paths.qa) / f"{sid}.json"):
            if p.is_file():
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("could not read %s: %s", p, exc)
                    continue
                for q in d.get("questions", []) or []:
                    if isinstance(q, dict) and q.get("qid"):
                        info[str(q["qid"])] = {"category": q.get("category", ""), "question": q.get("question", "")}
                info["_procedure_category"] = d.get("procedure_category", "")
                break
        self._qcache[sid] = info
        return info

    def qids(self, sid: str) -> list[str]:
        return sorted(k for k in self._questions(sid) if not k.startswith("_"))

    def qcategory(self, sid: str, qid: str) -> str:
        q = self._questions(sid).get(qid)
        return str(q.get("category") or "") if isinstance(q, dict) else ""


# ====================================================================== statistics
def cohen_kappa(pairs: list[tuple[Any, Any]]) -> Optional[float]:
    """Cohen's kappa for two raters over categorical labels; None when undefined."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po >= 1.0 - 1e-12 else None
    return round((po - pe) / (1.0 - pe), 3)


def group_stats(rows: list[dict]) -> dict:
    """Selection rate, correctness distribution and mean ratings over merged rows."""
    scored = [r for r in rows if r.get("qid")]
    n_sel = sum(1 for r in scored if r["selected"])
    corr = Counter(r["correctness"] or "unrated" for r in scored)
    return {
        "n_question_annotations": len(scored),
        "n_selected": n_sel,
        "selection_rate_pct": pct(n_sel, len(scored)),
        "n_edited": sum(1 for r in scored if r["edited"]),
        "correctness": dict(sorted(corr.items())),
        "correct_or_partial_pct": pct(corr["correct"] + corr["partial"], len(scored) - corr["unrated"]),
        "mean_ratings": {k: mean(r[k] for r in scored) for k in RATING_KEYS},
    }


def agreement_stats(anns: dict[tuple[str, str], SampleAnn], info: SampleInfo) -> dict:
    """Pairwise agreement on `selected` (and correctness) for samples seen by >= 2 annotators."""
    by_sample: dict[str, dict[str, SampleAnn]] = defaultdict(dict)
    for (annotator, sid), sa in anns.items():
        by_sample[sid][annotator] = sa
    shared = {sid: d for sid, d in by_sample.items() if len(d) >= 2}
    pair_sel: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    pair_corr: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    pair_samples: dict[tuple[str, str], set[str]] = defaultdict(set)
    disagreements: list[dict] = []
    for sid, d in shared.items():
        qids = set(info.qids(sid))
        for sa in d.values():
            qids |= set(sa.questions)
        for a, b in combinations(sorted(d), 2):
            sa, sb = d[a], d[b]
            pair_samples[(a, b)].add(sid)
            sel_a = {q: bool(sa.questions.get(q) and sa.questions[q].selected) for q in qids}
            sel_b = {q: bool(sb.questions.get(q) and sb.questions[q].selected) for q in qids}
            for q in sorted(qids):
                pair_sel[(a, b)].append((sel_a[q], sel_b[q]))
                ca = sa.questions[q].correctness if q in sa.questions else ""
                cb = sb.questions[q].correctness if q in sb.questions else ""
                if ca and cb:
                    pair_corr[(a, b)].append((ca, cb))
            if sel_a != sel_b:
                disagreements.append({"sample_id": sid, "annotator_a": a, "annotator_b": b,
                                      "selected_a": sorted(q for q, v in sel_a.items() if v),
                                      "selected_b": sorted(q for q, v in sel_b.items() if v)})
    pairs = []
    all_sel: list[tuple[bool, bool]] = []
    for (a, b), items in sorted(pair_sel.items()):
        agree = sum(1 for x, y in items if x == y)
        corr_items = pair_corr.get((a, b), [])
        pairs.append({
            "annotator_a": a, "annotator_b": b, "n_shared_samples": len(pair_samples[(a, b)]),
            "n_items": len(items), "selected_percent_agreement": pct(agree, len(items)),
            "selected_cohen_kappa": cohen_kappa(items),
            "correctness_n_items": len(corr_items),
            "correctness_percent_agreement": pct(sum(1 for x, y in corr_items if x == y), len(corr_items)),
            "correctness_cohen_kappa": cohen_kappa(corr_items),
        })
        all_sel.extend(items)
    kappas = [p["selected_cohen_kappa"] for p in pairs if p["selected_cohen_kappa"] is not None]
    return {
        "n_samples_multi_annotated": len(shared),
        "n_annotator_pairs": len(pairs),
        "pooled_selected_percent_agreement": pct(sum(1 for x, y in all_sel if x == y), len(all_sel)),
        "pooled_selected_cohen_kappa": cohen_kappa(all_sel),
        "mean_pairwise_selected_kappa": mean(kappas),
        "pairs": pairs,
        "n_samples_with_selection_disagreement": len({d["sample_id"] for d in disagreements}),
        "disagreements": disagreements[:500],
    }


def per_sample_rows(anns: dict[tuple[str, str], SampleAnn], info: SampleInfo) -> list[dict]:
    by_sample: dict[str, list[SampleAnn]] = defaultdict(list)
    for (_, sid), sa in anns.items():
        by_sample[sid].append(sa)
    out = []
    for sid in sorted(by_sample):
        sas = by_sample[sid]
        qids = set(info.qids(sid))
        for sa in sas:
            qids |= set(sa.questions)
        counts = {q: sum(1 for sa in sas if sa.questions.get(q) and sa.questions[q].selected) for q in sorted(qids)}
        n = len(sas)
        consensus = [q for q, c in counts.items() if c * 2 > n] if n > 1 else [q for q, c in counts.items() if c]
        flags = Counter(f for sa in sas for f in sa.flags)
        out.append({
            "sample_id": sid, "dataset": info.dataset(sid), "procedure_category": info.category(sid),
            "n_annotators": n, "n_done": sum(1 for sa in sas if sa.status == "done"),
            "annotators": ";".join(sorted(sa.annotator for sa in sas)),
            "selected_counts": json.dumps(counts, sort_keys=True), "n_selected_total": sum(counts.values()),
            "consensus_selected": ";".join(consensus),
            "flags": ";".join(f"{k}:{v}" for k, v in sorted(flags.items())),
            "not_suitable": int(flags.get("not_suitable", 0) > 0),
            "any_incorrect": int(any(q.correctness == "incorrect" for sa in sas for q in sa.questions.values())),
        })
    return out


# ====================================================================== merged rows
def merged_rows(anns: dict[tuple[str, str], SampleAnn], info: SampleInfo) -> list[dict]:
    rows: list[dict] = []
    for (annotator, sid), sa in sorted(anns.items()):
        qids = sorted(set(info.qids(sid)) | set(sa.questions)) or [""]
        for qid in qids:
            q = sa.questions.get(qid, QuestionAnn()) if qid else QuestionAnn()
            rows.append({
                "annotator": annotator, "sample_id": sid, "dataset": info.dataset(sid),
                "procedure_category": info.category(sid), "qid": qid or None,
                "category": info.qcategory(sid, qid) if qid else "",
                "annotated": qid in sa.questions,
                "selected": q.selected, "correctness": q.correctness,
                "relevance": q.relevance, "difficulty": q.difficulty, "agentic": q.agentic, "clarity": q.clarity,
                "edited": q.edited, "question_edit": q.question_edit, "answer_edit": q.answer_edit,
                "question_comment": q.comment,
                "sample_status": sa.status, "sample_flags": list(sa.flags), "sample_comment": sa.comment,
                "updated_at": sa.updated_at, "received_at": sa.received_at, "source": sa.source,
            })
    return rows


# ====================================================================== report
def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
    return str(v)


def write_summary_md(path: Path, summary: dict) -> None:
    s = summary
    lines = [f"# Annotation summary", "", f"Generated {s['generated_at']} from {s['inputs']['n_json_files']} JSON file(s) "
             f"and {s['inputs']['n_sheet_csvs']} sheet CSV(s). Only-done filter: {s['inputs']['only_done']}.", ""]
    o = s["overall"]
    lines += ["## Overall", "",
              md_table(["metric", "value"], [
                  ["annotators", o["n_annotators"]], ["sample annotations (annotator x sample)", o["n_sample_annotations"]],
                  ["distinct samples annotated", o["n_samples"]], ["marked complete", o["n_done"]],
                  ["question annotations", o["stats"]["n_question_annotations"]], ["questions selected", o["stats"]["n_selected"]],
                  ["selection rate %", fmt(o["stats"]["selection_rate_pct"])], ["edited questions/answers", o["stats"]["n_edited"]],
                  ["correct or partial % (of rated)", fmt(o["stats"]["correct_or_partial_pct"])],
                  ["mean relevance / difficulty / agentic / clarity",
                   " / ".join(fmt(o["stats"]["mean_ratings"][k]) for k in RATING_KEYS)],
                  ["samples flagged not_suitable", o["n_not_suitable"]],
              ]), ""]

    def stats_table(title: str, groups: dict) -> list[str]:
        rows = []
        for name, g in sorted(groups.items()):
            st = g["stats"]
            rows.append([name or "(unknown)", g["n_samples"], st["n_question_annotations"], st["n_selected"], fmt(st["selection_rate_pct"]),
                         st["correctness"].get("correct", 0), st["correctness"].get("partial", 0), st["correctness"].get("incorrect", 0),
                         st["correctness"].get("cannot_verify", 0), fmt(st["mean_ratings"]["relevance"]), fmt(st["mean_ratings"]["difficulty"]),
                         fmt(st["mean_ratings"]["agentic"]), fmt(st["mean_ratings"]["clarity"])])
        return [f"## {title}", "", md_table(["group", "samples", "q-annots", "selected", "sel %", "correct", "partial", "incorrect",
                                             "cannot verify", "relev", "diff", "agentic", "clarity"], rows), ""]

    lines += stats_table("Per dataset", s["per_dataset"])
    lines += stats_table("Per procedure category", s["per_procedure_category"])
    lines += stats_table("Per question category", s["per_question_category"])
    lines += stats_table("Per annotator", s["per_annotator"])

    a = s["agreement"]
    lines += ["## Inter-annotator agreement (selected for benchmark)", "",
              f"Samples seen by >= 2 annotators: {a['n_samples_multi_annotated']}; pooled percent agreement "
              f"{fmt(a['pooled_selected_percent_agreement'])}%, pooled Cohen kappa {fmt(a['pooled_selected_cohen_kappa'])}, "
              f"mean pairwise kappa {fmt(a['mean_pairwise_selected_kappa'])}; samples with a selection disagreement: "
              f"{a['n_samples_with_selection_disagreement']}.", ""]
    if a["pairs"]:
        lines += [md_table(["annotator A", "annotator B", "shared samples", "items", "selected agree %", "selected kappa",
                            "correctness items", "correctness agree %", "correctness kappa"],
                           [[p["annotator_a"], p["annotator_b"], p["n_shared_samples"], p["n_items"], fmt(p["selected_percent_agreement"]),
                             fmt(p["selected_cohen_kappa"]), p["correctness_n_items"], fmt(p["correctness_percent_agreement"]),
                             fmt(p["correctness_cohen_kappa"])] for p in a["pairs"]]), ""]
    if a["disagreements"]:
        lines += ["### Selection disagreements (first 50)", "",
                  md_table(["sample", "A", "A selected", "B", "B selected"],
                           [[d["sample_id"], d["annotator_a"], ",".join(d["selected_a"]) or "-", d["annotator_b"], ",".join(d["selected_b"]) or "-"]
                            for d in a["disagreements"][:50]]), ""]
    flagged = [r for r in s["per_sample"] if r["flags"]]
    if flagged:
        lines += ["## Flagged samples", "",
                  md_table(["sample", "dataset", "annotators", "flags", "consensus selected"],
                           [[r["sample_id"], r["dataset"], r["n_annotators"], r["flags"], r["consensus_selected"] or "-"] for r in flagged[:200]]), ""]
    act = summary.get("activity") or {}
    if act:
        lines += ["## Clinician activity", "",
                  "Active time counts only while the clinician works on a video (pauses after 2 min idle).", "",
                  md_table(["clinician", "videos done", "started", "selected", "edited", "active time", "min / done video",
                            "sessions", "first activity", "last activity"],
                           [[who, a["videos_done"], a["videos_started"], a["questions_selected"], a["questions_edited"],
                             f"{a['active_h']} h", fmt(a["avg_min_per_done_video"]), a["sessions"],
                             (a["first_activity"] or "-")[:16], (a["last_activity"] or "-")[:16]]
                            for who, a in act.items()]), ""]
    lines += ["## Files", "", "* merged.jsonl - one row per annotator x sample x question", "* per_sample.csv - selections per sample",
              "* summary.json - all numbers above in machine-readable form", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ====================================================================== main
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="pipeline config (default config/pipeline.yaml)")
    p.add_argument("--data-dir", default=None, help="data directory (default from config / $OPHBENCH_DATA_DIR)")
    p.add_argument("--annotations-dir", default=None, help="folder with annotation JSON files (default <data_dir>/annotations)")
    p.add_argument("--sheet-csv", action="append", default=[], help="CSV export of the Apps Script 'questions' sheet (repeatable)")
    p.add_argument("--samples-csv", action="append", default=[], help="CSV export of the Apps Script 'samples' sheet (repeatable)")
    p.add_argument("--only-done", action="store_true", help="ignore sample annotations not marked complete")
    p.add_argument("--out-dir", default=None, help="output folder (default <data_dir>/annotations)")
    p.add_argument("--ui-dir", default=None, help="UI folder for sample metadata (default <repo>/ui)")
    p.add_argument("-v", "--verbose", action="store_true")


def main(args: argparse.Namespace, cfg: Config) -> int:
    log = get_logger(STAGE, cfg, level=logging.DEBUG if args.verbose else logging.INFO)
    cfg.ensure_dirs()
    started = now_iso()
    ann_dir = Path(args.annotations_dir) if args.annotations_dir else Path(cfg.paths.annotations)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.paths.annotations)
    ui_dir = Path(args.ui_dir) if args.ui_dir else Path(cfg.paths.ui)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(p for p in ann_dir.glob("*.json") if p.name not in ("summary.json",)) if ann_dir.is_dir() else []
    log.info("annotations dir %s: %d JSON files; %d question CSVs; %d sample CSVs", ann_dir, len(json_files), len(args.sheet_csv), len(args.samples_csv))
    items = load_json_records(json_files, log)
    items += load_sheet_csvs([Path(p) for p in args.sheet_csv], [Path(p) for p in args.samples_csv], log)
    if not items:
        log.warning("no annotations found - nothing to merge (expected %s/*.json or --sheet-csv)", ann_dir)
    anns = dedupe_latest(items)
    if args.only_done:
        anns = {k: v for k, v in anns.items() if v.status == "done"}
    log.info("%d sample annotations after de-duplication (%d raw)", len(anns), len(items))

    info = SampleInfo(cfg, ui_dir, log)
    rows = merged_rows(anns, info)
    n_rows = write_jsonl(out_dir / "merged.jsonl", rows)

    def grouped(key) -> dict:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        return {k: {"n_samples": len({r["sample_id"] for r in v}), "stats": group_stats(v)} for k, v in groups.items()}

    per_sample = per_sample_rows(anns, info)
    summary = {
        "generated_at": now_iso(),
        "inputs": {"annotations_dir": str(ann_dir), "n_json_files": len(json_files), "json_files": [p.name for p in json_files],
                   "n_sheet_csvs": len(args.sheet_csv) + len(args.samples_csv), "only_done": bool(args.only_done)},
        "overall": {
            "n_annotators": len({a for a, _ in anns}), "n_sample_annotations": len(anns),
            "n_samples": len({s for _, s in anns}), "n_done": sum(1 for v in anns.values() if v.status == "done"),
            "n_not_suitable": sum(1 for r in per_sample if r["not_suitable"]),
            "stats": group_stats(rows),
        },
        "per_dataset": grouped(lambda r: r["dataset"]),
        "per_procedure_category": grouped(lambda r: r["procedure_category"]),
        "per_question_category": grouped(lambda r: r["category"]),
        "per_annotator": grouped(lambda r: r["annotator"]),
        "agreement": agreement_stats(anns, info),
        "activity": activity_stats(json_files, log),
        "per_sample": per_sample,
    }
    write_json(out_dir / "summary.json", summary)
    write_summary_md(out_dir / "summary.md", summary)
    if per_sample:
        with open(out_dir / "per_sample.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_sample[0].keys()))
            w.writeheader()
            w.writerows(per_sample)
    else:
        (out_dir / "per_sample.csv").write_text("sample_id\n", encoding="utf-8")

    o = summary["overall"]
    log.info("merged %d rows -> %s", n_rows, out_dir / "merged.jsonl")
    log.info("%d annotators, %d sample annotations (%d done), %d questions selected (%s%%); agreement over %d shared samples: kappa %s",
             o["n_annotators"], o["n_sample_annotations"], o["n_done"], o["stats"]["n_selected"], o["stats"]["selection_rate_pct"],
             summary["agreement"]["n_samples_multi_annotated"], summary["agreement"]["pooled_selected_cohen_kappa"])
    log.info("summary: %s", out_dir / "summary.md")
    record_run(cfg, STAGE, args, {"n_json_files": len(json_files), "n_sample_annotations": len(anns), "n_rows": n_rows,
                                  "n_selected": o["stats"]["n_selected"], "out_dir": str(out_dir)}, started_at=started)
    return 0


def _cli(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    args = p.parse_args(argv)
    if args.data_dir:
        os.environ["OPHBENCH_DATA_DIR"] = str(Path(args.data_dir).resolve())
    cfg = load_config(args.config)
    return main(args, cfg)


if __name__ == "__main__":
    sys.exit(_cli())
