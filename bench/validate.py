"""Validate stage (DESIGN.md §8): rule-based checks on the generated question sets.

Inputs
    data/qa/<sample_id>.json               QASet dicts written by the generate stage
    data/prepared/batch_<KKK>/<sample_id>/sample.json  probe duration (fallback: manifest ``duration_s``)
    data/sample/sample_manifest.jsonl      dataset / duration fallback

Outputs
    data/qa/qa_validated.jsonl             one row per QASet: ``QASet.to_dict()`` plus
                                           ``"validation": {"ok", "issues", "question_issues"}``
    data/qa/validation_report.json / .md   counts per issue type, per dataset and per category

Checks (hard issues make ``ok`` false)
    question_count            not exactly ``llm.questions_per_video`` (3) questions
    duplicate_qid             two questions share a qid
    duplicate_category        two questions share a category
    unknown_category / unknown_answer_type
    empty_question / empty_answer
    answer_in_question        the answer appears verbatim in the question text
    evidence_invalid          evidence entry is not {start_s, end_s} numbers with end >= start
    evidence_out_of_range     evidence outside [0, duration + 1] s (prepared probe duration)
    duplicate_question        token-Jaccard >= 0.6 between two questions of the same video
    mc_option_count           multiple_choice without 4-5 options
    mc_answer_not_in_options  multiple_choice answer references none of the options
    confidence_out_of_range   confidence is not a number in [0, 1]
Soft flags (recorded, never fail the item)
    low_confidence (< 0.5), no_evidence, options_on_non_mc, multiple_mc_questions,
    unknown_difficulty, unknown_agentic_skill, missing_rationale

Issue codes are ``<type>[:<detail>]``. ``question_issues[qid]`` holds a question's codes; the
set-level ``issues`` list holds set-level codes plus every question code as ``<qid>:<code>``.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from bench.batches import prepared_dir
from bench.config import Config
from bench.log import get_logger, now_iso, record_run
from bench.schema import AGENTIC_SKILLS, ANSWER_TYPES, DATASETS, DIFFICULTIES, QA_CATEGORIES, QAItem, QASet
from bench.util import iter_jsonl, jaccard, md_table, read_json, tokenize_simple, write_json, write_jsonl

STAGE = "validate"

DUPLICATE_JACCARD = 0.6
LOW_CONFIDENCE = 0.5
MC_MIN_OPTIONS = 4
MC_MAX_OPTIONS = 5
MIN_CONTAINMENT_CHARS = 4
EVIDENCE_SLACK_S = 1.0

HARD_ISSUES = frozenset({
    "question_count", "duplicate_qid", "duplicate_category", "unknown_category", "unknown_answer_type",
    "empty_question", "empty_answer", "answer_in_question", "evidence_invalid", "evidence_out_of_range",
    "duplicate_question", "mc_option_count", "mc_answer_not_in_options", "confidence_out_of_range",
    "load_error",
})
SOFT_ISSUES = frozenset({
    "low_confidence", "no_evidence", "options_on_non_mc", "multiple_mc_questions", "unknown_difficulty",
    "unknown_agentic_skill", "missing_rationale",
})

_QID_PREFIX_RE = re.compile(r"^q\d+:")
_OPTION_RE = re.compile(r"^\s*\(?([A-Za-z])\s*[\.\):\-]\s*(.*?)\s*$")
_ANSWER_LETTER_RE = re.compile(r"^\s*\(?([A-Za-z])(?:\s*[\.\):\-]\s*|\)\s*|\s*$)(.*)$")
_PUNCT_STRIP = " \t\r\n.,;:!?\"'()[]{}"


# ------------------------------------------------------------------------------------ helpers
def issue_type(code: str) -> str:
    """``'q2:mc_option_count:3'`` -> ``'mc_option_count'``."""
    code = _QID_PREFIX_RE.sub("", code or "")
    return code.split(":", 1)[0]


def is_hard(code: str) -> bool:
    return issue_type(code) in HARD_ISSUES


def _norm(text: Any) -> str:
    """Lower-case, collapse whitespace, strip surrounding punctuation."""
    return " ".join(str(text or "").lower().split()).strip(_PUNCT_STRIP)


def parse_option(option: str) -> tuple[Optional[str], str]:
    """Split ``'B. Hydrodissection'`` / ``'(b) text'`` / ``'B) text'`` into ``('B', 'text')``."""
    m = _OPTION_RE.match(str(option or ""))
    if m:
        return m.group(1).upper(), m.group(2)
    return None, str(option or "").strip()


def mc_answer_matches(answer: str, options: list[str]) -> bool:
    """True when the answer references one option by letter (``'B'``, ``'B.'``, ``'(B) text'``) or text."""
    letters: dict[str, str] = {}
    for i, opt in enumerate(options):
        letter, text = parse_option(opt)
        letters[letter or chr(ord("A") + i)] = _norm(text)
    ans = str(answer or "")
    m = _ANSWER_LETTER_RE.match(ans)
    if m and m.group(1).upper() in letters:
        return True
    a = _norm(ans)
    if not a:
        return False
    for text in letters.values():
        if not text:
            continue
        if a == text:
            return True
        if len(a) >= 3 and len(text) >= 3 and (a in text or text in a):
            return True
    return False


def dataset_of(sample_id: str) -> str:
    ds = (sample_id or "").split("__", 1)[0]
    return ds if ds in DATASETS else (ds or "unknown")


# ------------------------------------------------------------------------------------ core checks
@dataclass
class ValidationResult:
    ok: bool = True
    issues: list[str] = field(default_factory=list)
    question_issues: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "issues": list(self.issues),
            "question_issues": {k: list(v) for k, v in self.question_issues.items()},
        }


def _coerce_float(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _check_evidence(q: QAItem, duration_s: Optional[float]) -> list[str]:
    codes: list[str] = []
    ev_list = q.evidence_timestamps
    if ev_list is None:
        ev_list = []
    if not isinstance(ev_list, list):
        return ["evidence_invalid:not_a_list"]
    if not ev_list:
        codes.append("no_evidence")
        return codes
    invalid = 0
    out_of_range: list[str] = []
    limit = (duration_s + EVIDENCE_SLACK_S) if duration_s is not None and duration_s > 0 else None
    for ev in ev_list:
        if not isinstance(ev, dict):
            invalid += 1
            continue
        s, e = _coerce_float(ev.get("start_s")), _coerce_float(ev.get("end_s"))
        if s is None or e is None or e < s:
            invalid += 1
            continue
        if s < 0 or (limit is not None and e > limit):
            out_of_range.append(f"{s:g}-{e:g}s")
    if invalid:
        codes.append(f"evidence_invalid:{invalid}")
    if out_of_range:
        detail = out_of_range[0] + (f"(+{len(out_of_range) - 1})" if len(out_of_range) > 1 else "")
        if limit is not None:
            detail += f">{limit:g}s"
        codes.append(f"evidence_out_of_range:{detail}")
    return codes


def check_question(q: QAItem, duration_s: Optional[float]) -> list[str]:
    """Per-question checks (everything except cross-question duplicates / counts)."""
    codes: list[str] = []
    question, answer = _norm(q.question), _norm(q.answer)
    if q.category not in QA_CATEGORIES:
        codes.append(f"unknown_category:{q.category}")
    if q.answer_type not in ANSWER_TYPES:
        codes.append(f"unknown_answer_type:{q.answer_type}")
    if q.difficulty not in DIFFICULTIES:
        codes.append(f"unknown_difficulty:{q.difficulty}")
    for skill in q.agentic_skills or []:
        if skill not in AGENTIC_SKILLS:
            codes.append(f"unknown_agentic_skill:{skill}")
    if not question:
        codes.append("empty_question")
    if not answer:
        codes.append("empty_answer")
    elif question and q.answer_type != "boolean" and len(answer) >= MIN_CONTAINMENT_CHARS and answer in question:
        codes.append("answer_in_question")
    if not _norm(q.answer_rationale):
        codes.append("missing_rationale")
    codes.extend(_check_evidence(q, duration_s))

    options = [str(o) for o in (q.options or []) if str(o).strip()]
    if q.answer_type == "multiple_choice":
        if not (MC_MIN_OPTIONS <= len(options) <= MC_MAX_OPTIONS):
            codes.append(f"mc_option_count:{len(options)}")
        if answer and options and not mc_answer_matches(q.answer, options):
            codes.append("mc_answer_not_in_options")
    elif options:
        codes.append(f"options_on_non_mc:{len(options)}")

    conf = _coerce_float(q.confidence)
    if conf is None:
        codes.append(f"confidence_out_of_range:{q.confidence!r}")
    elif conf < 0 or conf > 1:
        codes.append(f"confidence_out_of_range:{conf:g}")
    elif conf < LOW_CONFIDENCE:
        codes.append(f"low_confidence:{conf:g}")
    return codes


def validate_qaset(qs: QASet, duration_s: Optional[float], expected_n: int = 3) -> ValidationResult:
    """Run every check on one QASet; pure function (no I/O)."""
    res = ValidationResult()
    set_issues: list[str] = []
    per_q: dict[str, list[str]] = defaultdict(list)
    questions = list(qs.questions or [])

    if len(questions) != expected_n:
        set_issues.append(f"question_count:{len(questions)}")

    qid_counts = Counter(q.qid for q in questions)
    for qid, n in qid_counts.items():
        if n > 1:
            set_issues.append(f"duplicate_qid:{qid}")

    seen_cat: dict[str, str] = {}
    for q in questions:
        if q.category in seen_cat:
            set_issues.append(f"duplicate_category:{q.category}")
            per_q[q.qid].append(f"duplicate_category:{q.category}")
        else:
            seen_cat[q.category] = q.qid

    if sum(1 for q in questions if q.answer_type == "multiple_choice") > 1:
        set_issues.append("multiple_mc_questions")

    for q in questions:
        per_q[q.qid].extend(check_question(q, duration_s))

    tokens = [tokenize_simple(q.question) for q in questions]
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if tokens[i] and tokens[j] and jaccard(tokens[i], tokens[j]) >= DUPLICATE_JACCARD:
                set_issues.append(f"duplicate_question:{questions[i].qid}~{questions[j].qid}")
                per_q[questions[j].qid].append(f"duplicate_question:{questions[i].qid}")

    issues = list(set_issues)
    for q in questions:
        for code in per_q.get(q.qid, []):
            issues.append(f"{q.qid}:{code}")
    # keep every qid as a key (empty list = clean question) for a stable shape
    res.question_issues = {q.qid: list(per_q.get(q.qid, [])) for q in questions}
    res.issues = issues
    res.ok = not any(is_hard(c) for c in issues)
    return res


# ------------------------------------------------------------------------------------ inputs
def discover_qa_files(qa_dir: Path) -> list[Path]:
    """QA files are ``<dataset>__<video>.json``; report/aux files never contain ``__``."""
    if not qa_dir.exists():
        return []
    return sorted(p for p in qa_dir.glob("*.json") if "__" in p.stem and p.is_file())


def load_manifest_info(cfg: Config) -> dict[str, dict]:
    """sample_id -> {dataset, duration_s} from the sample manifest (empty when missing)."""
    path = Path(cfg.paths.sample) / "sample_manifest.jsonl"
    info: dict[str, dict] = {}
    for row in iter_jsonl(path):
        sid = row.get("sample_id")
        if sid:
            info[sid] = {"dataset": row.get("dataset"), "duration_s": row.get("duration_s")}
    return info


def prepared_duration(cfg: Config, sample_id: str) -> Optional[float]:
    """Probe duration from data/prepared/batch_<KKK>/<id>/sample.json, or None."""
    path = prepared_dir(cfg, sample_id) / "sample.json"
    if not path.exists():
        return None
    try:
        d = read_json(path)
        dur = _coerce_float((d.get("probe") or {}).get("duration_s"))
        return dur if dur and dur > 0 else None
    except Exception:
        return None


def resolve_duration(cfg: Config, sample_id: str, manifest: dict[str, dict]) -> Optional[float]:
    dur = prepared_duration(cfg, sample_id)
    if dur is None:
        dur = _coerce_float((manifest.get(sample_id) or {}).get("duration_s"))
    return dur if dur and dur > 0 else None


# ------------------------------------------------------------------------------------ report
def _top_issues(counter: Counter, n: int = 4) -> str:
    return ", ".join(f"{k} x{v}" for k, v in counter.most_common(n)) or "-"


def build_report(rows: list[dict], load_errors: list[dict], expected_n: int, paths: dict[str, str]) -> dict:
    issue_counts: Counter = Counter()
    hard_counts: Counter = Counter()
    soft_counts: Counter = Counter()
    per_dataset: dict[str, dict] = defaultdict(lambda: {"n": 0, "ok": 0, "with_issues": 0, "issues": Counter()})
    per_category: dict[str, dict] = defaultdict(lambda: {"n_questions": 0, "with_hard_issues": 0, "issues": Counter()})
    answer_types: Counter = Counter()
    difficulties: Counter = Counter()
    confidences: list[float] = []
    examples: dict[str, list[str]] = defaultdict(list)
    n_questions = 0

    for row in rows:
        v = row.get("validation") or {}
        ds = dataset_of(row.get("sample_id", ""))
        pd = per_dataset[ds]
        pd["n"] += 1
        pd["ok" if v.get("ok") else "with_issues"] += 1
        for code in v.get("issues", []):
            t = issue_type(code)
            issue_counts[t] += 1
            pd["issues"][t] += 1
            (hard_counts if t in HARD_ISSUES else soft_counts)[t] += 1
            if len(examples[t]) < 5:
                examples[t].append(f"{row.get('sample_id')} {code}")
        qi = v.get("question_issues", {})
        for q in row.get("questions", []):
            n_questions += 1
            pc = per_category[str(q.get("category"))]
            pc["n_questions"] += 1
            codes = qi.get(q.get("qid"), [])
            if any(is_hard(c) for c in codes):
                pc["with_hard_issues"] += 1
            for c in codes:
                pc["issues"][issue_type(c)] += 1
            answer_types[str(q.get("answer_type"))] += 1
            difficulties[str(q.get("difficulty"))] += 1
            conf = _coerce_float(q.get("confidence"))
            if conf is not None:
                confidences.append(conf)

    for e in load_errors:
        ds = dataset_of(e.get("sample_id", ""))
        per_dataset[ds]["issues"]["load_error"] += 1
        issue_counts["load_error"] += 1
        hard_counts["load_error"] += 1
        if len(examples["load_error"]) < 5:
            examples["load_error"].append(f"{e.get('sample_id')} {e.get('error')}")

    n_ok = sum(1 for r in rows if (r.get("validation") or {}).get("ok"))
    return {
        "generated_at": now_iso(),
        "expected_questions_per_video": expected_n,
        "n_qa_sets": len(rows),
        "n_ok": n_ok,
        "n_with_issues": len(rows) - n_ok,
        "n_load_errors": len(load_errors),
        "n_questions": n_questions,
        "issue_counts": dict(issue_counts.most_common()),
        "hard_issue_counts": dict(hard_counts.most_common()),
        "soft_flag_counts": dict(soft_counts.most_common()),
        "per_dataset": {
            ds: {**d, "issues": dict(d["issues"].most_common())} for ds, d in sorted(per_dataset.items())
        },
        "per_category": {
            c: {**d, "issues": dict(d["issues"].most_common())} for c, d in sorted(per_category.items())
        },
        "answer_types": dict(answer_types.most_common()),
        "difficulties": dict(difficulties.most_common()),
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "n_low_confidence": sum(1 for c in confidences if c < LOW_CONFIDENCE),
        "examples": dict(examples),
        "load_errors": load_errors,
        "paths": paths,
    }


def report_markdown(rep: dict) -> str:
    n = rep["n_qa_sets"] or 1
    lines = [
        "# QA validation report",
        "",
        f"Generated: {rep['generated_at']}  ",
        f"QA sets: **{rep['n_qa_sets']}** - ok: **{rep['n_ok']}** ({100.0 * rep['n_ok'] / n:.1f}%), "
        f"with issues: **{rep['n_with_issues']}**, load errors: {rep['n_load_errors']}  ",
        f"Questions: {rep['n_questions']} (expected {rep['expected_questions_per_video']} per video); "
        f"mean confidence {rep['mean_confidence']}; low-confidence questions: {rep['n_low_confidence']}",
        "",
        "## Issues by type",
        "",
        md_table(["issue", "severity", "count"],
                 [[k, "hard" if k in HARD_ISSUES else "soft", v] for k, v in rep["issue_counts"].items()]) or "_none_",
        "",
        "## Per dataset",
        "",
        md_table(["dataset", "sets", "ok", "with issues", "top issues"],
                 [[ds, d["n"], d["ok"], d["with_issues"], _top_issues(Counter(d["issues"]))]
                  for ds, d in rep["per_dataset"].items()]),
        "",
        "## Per question category",
        "",
        md_table(["category", "questions", "with hard issues", "top issues"],
                 [[c, d["n_questions"], d["with_hard_issues"], _top_issues(Counter(d["issues"]))]
                  for c, d in rep["per_category"].items()]),
        "",
        "## Answer types",
        "",
        md_table(["answer_type", "count"], [[k, v] for k, v in rep["answer_types"].items()]),
        "",
        "## Difficulty",
        "",
        md_table(["difficulty", "count"], [[k, v] for k, v in rep["difficulties"].items()]),
        "",
        "## Examples (up to 5 per issue type)",
        "",
    ]
    for t, exs in rep["examples"].items():
        lines.append(f"* `{t}`: " + "; ".join(f"`{e}`" for e in exs))
    if not rep["examples"]:
        lines.append("_none_")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ stage entry
def add_args(p: argparse.ArgumentParser) -> None:
    """``validate`` takes no stage-specific options."""


def main(args: argparse.Namespace, cfg: Config) -> int:
    cfg.ensure_dirs()
    log = get_logger(STAGE, cfg)
    started = now_iso()
    qa_dir = Path(cfg.paths.qa)
    expected_n = int(cfg.get("llm.questions_per_video", 3) or 3)
    manifest = load_manifest_info(cfg)
    files = discover_qa_files(qa_dir)
    log.info("validating %d QA file(s) under %s (expected %d questions per video)", len(files), qa_dir, expected_n)
    if not files:
        log.warning("no QA files found in %s - run `generate` first (writing empty outputs)", qa_dir)

    rows: list[dict] = []
    load_errors: list[dict] = []
    n_unknown_duration = 0
    for path in files:
        sid = path.stem
        try:
            qs = QASet.from_dict(read_json(path))
        except Exception as e:  # malformed file: report it, keep going
            log.warning("%s: cannot load QASet (%s: %s) - skipped", path.name, type(e).__name__, e)
            load_errors.append({"sample_id": sid, "file": str(path), "error": f"{type(e).__name__}: {e}"})
            continue
        if qs.sample_id != sid:
            log.warning("%s: sample_id inside file is %r (using file stem)", path.name, qs.sample_id)
            qs.sample_id = sid
        duration = resolve_duration(cfg, sid, manifest)
        if duration is None:
            n_unknown_duration += 1
            log.debug("%s: duration unknown - evidence range check skipped", sid)
        result = validate_qaset(qs, duration, expected_n)
        row = qs.to_dict()
        row["validation"] = result.to_dict()
        rows.append(row)
        if not result.ok:
            log.info("%s: %d issue(s): %s", sid, len(result.issues), ", ".join(result.issues[:6]))

    out_jsonl = qa_dir / "qa_validated.jsonl"
    out_json = qa_dir / "validation_report.json"
    out_md = qa_dir / "validation_report.md"
    write_jsonl(out_jsonl, rows)
    report = build_report(rows, load_errors, expected_n,
                          {"qa_validated": str(out_jsonl), "report_json": str(out_json), "report_md": str(out_md)})
    write_json(out_json, report)
    out_md.write_text(report_markdown(report), encoding="utf-8")

    summary = {
        "n_qa_sets": report["n_qa_sets"], "n_ok": report["n_ok"], "n_with_issues": report["n_with_issues"],
        "n_load_errors": report["n_load_errors"], "n_questions": report["n_questions"],
        "n_unknown_duration": n_unknown_duration, "hard_issue_counts": report["hard_issue_counts"],
        "soft_flag_counts": report["soft_flag_counts"], "outputs": report["paths"],
    }
    log.info("validated %d set(s): %d ok, %d with issues, %d load error(s); hard issues: %s; soft flags: %s",
             report["n_qa_sets"], report["n_ok"], report["n_with_issues"], report["n_load_errors"],
             report["hard_issue_counts"] or "none", report["soft_flag_counts"] or "none")
    log.info("wrote %s, %s, %s", out_jsonl, out_json, out_md)
    record_run(cfg, STAGE, args, summary, started_at=started)
    return 0
