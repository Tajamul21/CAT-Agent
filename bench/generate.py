"""Generation stage (DESIGN.md §7): prompts from prepared samples -> gateway -> ``data/qa/<id>.json``.

Resumable: samples whose ``data/qa/<sample_id>.json`` exists are skipped unless ``--force``.  Failed or
unparsable samples leave no QA file, so re-running retries them.  Per-sample artefacts:

* ``data/qa/raw/<id>.request.json``  - the request with base64 images redacted
* ``data/qa/raw/<id>.response.json`` - the final response JSON, text, usage, request id
* ``data/qa/<id>.json``              - ``QASet.to_dict()`` with provenance
* ``data/qa/generation_log.jsonl``   - one ``GenerationLogEntry`` per attempt
* ``data/qa/qa_all.jsonl``           - rebuilt from all QA files at the end of every run
* ``data/qa/dryrun/<id>.json``       - ``--dry-run``: text prompt + frame list, no API call
"""
from __future__ import annotations

import argparse
import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from bench.batches import batch_size, order_samples, prepared_dir, select_batch
from bench.config import Config
from bench.llm_client import EFFORTS, ROUTES, GatewayClient, LLMResult
from bench.log import JsonlWriter, get_logger, now_iso, record_run
from bench.prompts import build_messages, frame_list, prompt_fingerprint, prompt_version, user_text
from bench.schema import QA_OUTPUT_SCHEMA, GenerationLogEntry, PreparedSample, QASet, SampleRecord
from bench.util import iter_jsonl, read_json, write_json, write_jsonl

STAGE = "generate"


# ------------------------------------------------------------------------------------------ CLI
def add_args(p: argparse.ArgumentParser) -> None:
    """Options of ``./ophbench generate``."""
    p.add_argument("--concurrency", type=int, default=None, help="parallel API calls (default: llm.concurrency)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N pending samples this run (picked round-robin across datasets)")
    p.add_argument("--ids", default=None, help="comma-separated sample ids")
    p.add_argument("--datasets", default=None, help="comma-separated dataset names")
    p.add_argument("--dry-run", action="store_true", help="write data/qa/dryrun/<id>.json prompts; no API calls")
    p.add_argument("--force", action="store_true", help="regenerate samples that already have a QA file")
    p.add_argument("--route", choices=list(ROUTES), default=None, help="override llm.route")
    p.add_argument("--effort", choices=list(EFFORTS), default=None, help="override llm.reasoning_effort")
    p.add_argument("--max-tokens", type=int, default=None, help="override llm.max_completion_tokens")
    p.add_argument("--batch", type=int, default=None,
                   help="1-based annotation batch to generate (canonical benchmark order, DESIGN.md 13.1)")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="samples per batch (default: config `batches.size`, 100)")


def _csv_list(v: Any) -> Optional[list[str]]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        items = [str(x) for x in v]
    else:
        items = str(v).split(",")
    out = [s.strip() for s in items if s and s.strip()]
    return out or None


# ------------------------------------------------------------------------------------------ work
@dataclass
class Options:
    route: str
    effort: str
    max_tokens: int
    concurrency: int
    force: bool
    dry_run: bool
    model: str
    prompt_version: str
    prompt_sha: str


@dataclass
class WorkItem:
    record: SampleRecord
    sample_dir: Path
    sample_json: Path


def load_manifest(cfg: Config, log: logging.Logger) -> Optional[list[SampleRecord]]:
    path = Path(cfg.paths.sample) / "sample_manifest.jsonl"
    if not path.exists():
        log.error("sample manifest not found: %s (run `./ophbench sample` first)", path)
        return None
    records: list[SampleRecord] = []
    for i, row in enumerate(iter_jsonl(path)):
        try:
            records.append(SampleRecord.from_dict(row))
        except (KeyError, TypeError) as e:
            log.warning("manifest line %d unreadable (%s) - skipped", i + 1, e)
    return records


def _round_robin(items: list[WorkItem]) -> list[WorkItem]:
    """Interleave datasets so that ``--limit N`` touches every dataset (order is deterministic)."""
    groups: dict[str, list[WorkItem]] = {}
    for it in items:
        groups.setdefault(it.record.dataset, []).append(it)
    out: list[WorkItem] = []
    queues = [list(g) for g in groups.values()]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def select_work(records: list[SampleRecord], cfg: Config, *, ids: Optional[list[str]], datasets: Optional[list[str]],
                limit: Optional[int], force: bool, log: logging.Logger) -> tuple[list[WorkItem], dict[str, int]]:
    """Manifest ∩ prepared, minus already generated (unless force), filtered and limited."""
    counts = {"manifest": len(records), "filtered": 0, "unprepared": 0, "skipped_existing": 0, "selected": 0}
    qa_dir = Path(cfg.paths.qa)
    id_set = set(ids) if ids else None
    ds_set = set(datasets) if datasets else None
    items: list[WorkItem] = []
    for rec in records:
        if id_set and rec.sample_id not in id_set:
            continue
        if ds_set and rec.dataset not in ds_set:
            continue
        counts["filtered"] += 1
        sdir = prepared_dir(cfg, rec.sample_id)
        sjson = sdir / "sample.json"
        if not sjson.exists():
            counts["unprepared"] += 1
            # one line per sample only when the user asked for it explicitly; otherwise a summary below
            (log.warning if id_set else log.debug)("%s: not prepared (%s missing) - skipped", rec.sample_id, sjson)
            continue
        if not force and (qa_dir / f"{rec.sample_id}.json").exists():
            counts["skipped_existing"] += 1
            continue
        items.append(WorkItem(record=rec, sample_dir=sdir, sample_json=sjson))
    if counts["unprepared"] and not id_set:
        log.warning("%d of %d manifest sample(s) are not prepared yet and were skipped (run `./ophbench prepare`)",
                    counts["unprepared"], counts["filtered"])
    if id_set:
        missing = sorted(id_set - {r.sample_id for r in records})
        if missing:
            log.warning("%d requested id(s) not in the manifest: %s", len(missing), ", ".join(missing[:10]))
    if limit is not None and limit >= 0:
        items = _round_robin(items)[:limit]
    counts["selected"] = len(items)
    return items, counts


# ------------------------------------------------------------------------------------------ per sample
def _provenance(res: LLMResult, opts: Options, n_frames: int, generated_at: str) -> dict:
    return {
        "model": res.model or opts.model,
        "route": res.route,
        "effort": res.effort_applied or opts.effort,
        "effort_requested": res.effort_requested,
        "n_frames": n_frames,
        "usage": res.usage,
        "latency_s": round(res.latency_s, 2),
        "generated_at": generated_at,
        "prompt_version": opts.prompt_version,
        "prompt_sha": opts.prompt_sha,
        "attempts": res.attempts,
        "request_id": res.request_id,
        "repaired": res.repaired,
        "schema_issues": res.schema_issues,
        "finish_note": res.finish_note,
        "est_cost_usd": res.est_cost_usd,
    }


def generate_one(item: WorkItem, client: GatewayClient, cfg: Config, opts: Options, log: logging.Logger) -> GenerationLogEntry:
    """Build the prompt, call the gateway, persist artefacts; never raises (errors become log entries)."""
    sid = item.record.sample_id
    started = now_iso()
    t0 = time.monotonic()
    qa_dir = Path(cfg.paths.qa)
    raw_dir = qa_dir / "raw"
    entry = GenerationLogEntry(sample_id=sid, model=opts.model, route=opts.route, reasoning_effort=opts.effort,
                               n_frames=0, started_at=started, status="error")
    try:
        prepared = PreparedSample.from_dict(read_json(item.sample_json))
        system_text, parts = build_messages(prepared, cfg, sample_dir=item.sample_dir, log=log)
        n_frames = sum(1 for p in parts if p.get("type") == "image")
        entry.n_frames = n_frames
        if n_frames == 0:
            raise RuntimeError("no frame images available (run prepare for this sample)")
        req = client.build_request(parts, system=system_text, schema=QA_OUTPUT_SCHEMA, schema_name="qa_output",
                                   route=opts.route, reasoning_effort=opts.effort, max_tokens=opts.max_tokens)
        write_json(raw_dir / f"{sid}.request.json", {"sample_id": sid, "started_at": started, **req.redacted()})
        res = client.send(req)
        finished = now_iso()
        write_json(raw_dir / f"{sid}.response.json", {
            "sample_id": sid, "finished_at": finished, "request_id": res.request_id, "route": res.route,
            "model": res.model, "effort_requested": res.effort_requested, "effort_applied": res.effort_applied,
            "attempts": res.attempts, "latency_s": round(res.latency_s, 2), "usage": res.usage,
            "parse_error": res.parse_error, "schema_issues": res.schema_issues, "repaired": res.repaired,
            "finish_note": res.finish_note, "refusal": res.refusal, "text": res.text, "raw": res.raw,
        })
        entry.reasoning_effort = res.effort_applied or opts.effort
        entry.route = res.route
        entry.model = res.model or opts.model
        entry.prompt_tokens = res.usage.get("prompt_tokens")
        entry.completion_tokens = res.usage.get("completion_tokens")
        entry.reasoning_tokens = res.usage.get("reasoning_tokens")
        entry.latency_s = round(res.latency_s, 2)
        entry.request_id = res.request_id
        entry.est_cost_usd = res.est_cost_usd
        entry.attempts = res.attempts
        if res.parsed is None:
            entry.status = "parse_error"
            entry.error = (res.refusal and f"refusal: {res.refusal}") or res.parse_error or "unparsable output"
        else:
            qaset = QASet.from_llm_json(sid, res.parsed, _provenance(res, opts, n_frames, finished))
            write_json(qa_dir / f"{sid}.json", qaset.to_dict())
            entry.status = "ok"
            if len(qaset.questions) != int(cfg.get("llm.questions_per_video", 3) or 3):
                entry.error = f"expected {cfg.get('llm.questions_per_video', 3)} questions, got {len(qaset.questions)}"
            if res.schema_issues:
                entry.error = (entry.error + "; " if entry.error else "") + f"schema issues: {res.schema_issues[:3]}"
    except Exception as e:  # noqa: BLE001 - one bad sample must not kill the stage
        entry.status = "error"
        entry.error = f"{type(e).__name__}: {str(e)[:500]}"
        if entry.latency_s is None:
            entry.latency_s = round(time.monotonic() - t0, 2)
    entry.finished_at = now_iso()
    return entry


def dry_run_one(item: WorkItem, cfg: Config, opts: Options, log: logging.Logger) -> GenerationLogEntry:
    """Write ``data/qa/dryrun/<id>.json`` with the rendered text prompt and the frame list."""
    sid = item.record.sample_id
    started = now_iso()
    entry = GenerationLogEntry(sample_id=sid, model=opts.model, route=opts.route, reasoning_effort=opts.effort,
                               n_frames=0, status="dry_run", started_at=started)
    try:
        prepared = PreparedSample.from_dict(read_json(item.sample_json))
        system_text, parts = build_messages(prepared, cfg, sample_dir=item.sample_dir, log=log)
        frames = frame_list(parts)
        text = user_text(parts)
        entry.n_frames = len(frames)
        write_json(Path(cfg.paths.qa) / "dryrun" / f"{sid}.json", {
            "sample_id": sid, "dataset": item.record.dataset, "route": opts.route, "effort": opts.effort,
            "model": opts.model, "max_tokens": opts.max_tokens, "prompt_version": opts.prompt_version,
            "prompt_sha": opts.prompt_sha, "n_frames": len(frames), "n_text_chars": len(system_text) + len(text),
            "system": system_text, "user_text": text, "frames": frames, "generated_at": started,
        })
    except Exception as e:  # noqa: BLE001
        entry.status = "error"
        entry.error = f"{type(e).__name__}: {str(e)[:500]}"
    entry.finished_at = now_iso()
    return entry


# ------------------------------------------------------------------------------------------ outputs
def rebuild_qa_all(cfg: Config, log: logging.Logger) -> int:
    """Rebuild ``data/qa/qa_all.jsonl`` from every ``data/qa/<sample_id>.json`` (sorted by sample id)."""
    qa_dir = Path(cfg.paths.qa)
    rows: list[dict] = []
    for p in sorted(qa_dir.glob("*.json")):
        try:
            d = read_json(p)
        except (OSError, ValueError) as e:
            log.warning("unreadable QA file %s: %s", p.name, e)
            continue
        if isinstance(d, dict) and "sample_id" in d and "questions" in d:
            rows.append(d)
    n = write_jsonl(qa_dir / "qa_all.jsonl", rows)
    log.info("qa_all.jsonl rebuilt: %d QA sets", n)
    return n


def summarise(entries: list[GenerationLogEntry], counts: dict[str, int], opts: Options, elapsed_s: float,
              n_qa_all: int) -> dict:
    by_status: dict[str, int] = {}
    for e in entries:
        by_status[e.status] = by_status.get(e.status, 0) + 1
    tokens = {
        "prompt": sum(e.prompt_tokens or 0 for e in entries),
        "completion": sum(e.completion_tokens or 0 for e in entries),
        "reasoning": sum(e.reasoning_tokens or 0 for e in entries),
    }
    costs = [e.est_cost_usd for e in entries if e.est_cost_usd is not None]
    lat = [e.latency_s for e in entries if e.latency_s is not None and e.status in ("ok", "parse_error")]
    return {
        "route": opts.route, "effort": opts.effort, "model": opts.model, "max_tokens": opts.max_tokens,
        "concurrency": opts.concurrency, "dry_run": opts.dry_run, "prompt_version": opts.prompt_version,
        "prompt_sha": opts.prompt_sha,
        "counts": {**counts, "ok": by_status.get("ok", 0), "parse_error": by_status.get("parse_error", 0),
                   "error": by_status.get("error", 0), "dry_run": by_status.get("dry_run", 0)},
        "tokens": tokens,
        "est_cost_usd": round(sum(costs), 4) if costs else None,
        "mean_latency_s": round(statistics.fmean(lat), 1) if lat else None,
        "elapsed_s": round(elapsed_s, 1),
        "qa_all": n_qa_all,
    }


# ------------------------------------------------------------------------------------------ main
def main(args: argparse.Namespace, cfg: Config) -> int:
    """Entry point for ``./ophbench generate``; returns 0 unless the stage could not run or every sample failed."""
    cfg.ensure_dirs()
    log = get_logger(STAGE, cfg)
    started = now_iso()
    t_start = time.monotonic()
    llm = cfg.section("llm")
    dry_run = bool(getattr(args, "dry_run", False))
    opts = Options(
        route=getattr(args, "route", None) or str(llm.get("route", "responses")),
        effort=getattr(args, "effort", None) or str(llm.get("reasoning_effort", "xhigh")),
        max_tokens=int(getattr(args, "max_tokens", None) or llm.get("max_completion_tokens", 32000)),
        concurrency=max(1, int(getattr(args, "concurrency", None) or llm.get("concurrency", 4))),
        force=bool(getattr(args, "force", False)),
        dry_run=dry_run,
        model=str(llm.get("model", "openai/gpt-6-astra")),
        prompt_version=prompt_version(cfg),
        prompt_sha="",
    )
    try:
        opts.prompt_sha = prompt_fingerprint(cfg)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    log.info("generate: model=%s route=%s effort=%s max_tokens=%d concurrency=%d dry_run=%s force=%s prompt=%s/%s",
             opts.model, opts.route, opts.effort, opts.max_tokens, opts.concurrency, dry_run, opts.force,
             opts.prompt_version, opts.prompt_sha)
    if opts.route == "rest_chat" and not dry_run:
        log.error("route rest_chat is text-only on this gateway and cannot carry frames; use --route responses")
        return 1

    records = load_manifest(cfg, log)
    if records is None:
        return 1
    batch = getattr(args, "batch", None)
    if batch is not None:
        size = batch_size(cfg, getattr(args, "batch_size", None))
        try:
            records = select_batch(order_samples(records), int(batch), size)
        except ValueError as e:
            log.error("%s", e)
            return 1
        log.info("batch %d (size %d): %d sample(s) in this batch", batch, size, len(records))
        if not records:
            log.warning("batch %d is empty", batch)
    items, counts = select_work(records, cfg, ids=_csv_list(getattr(args, "ids", None)),
                                datasets=_csv_list(getattr(args, "datasets", None)),
                                limit=getattr(args, "limit", None), force=opts.force, log=log)
    log.info("manifest=%d filtered=%d unprepared=%d already_done=%d selected=%d", counts["manifest"],
             counts["filtered"], counts["unprepared"], counts["skipped_existing"], counts["selected"])

    entries: list[GenerationLogEntry] = []
    interrupted = False
    if dry_run:
        for it in items:
            e = dry_run_one(it, cfg, opts, log)
            entries.append(e)
            log.info("dry-run %s: %s (%d frames)%s", e.sample_id, e.status, e.n_frames, f" {e.error}" if e.error else "")
        dry_log = JsonlWriter(Path(cfg.paths.qa) / "dryrun" / "dryrun_log.jsonl")
        for e in entries:
            dry_log.write(e.to_dict())
        n_all = 0
    else:
        if not items:
            log.info("nothing to do")
        else:
            try:
                client = GatewayClient(cfg, log=log)
            except (RuntimeError, ValueError) as e:
                log.error("cannot create gateway client: %s", e)
                return 1
            try:
                entries, interrupted = _run_pool(items, client, cfg, opts, log)
            finally:
                client.close()
        n_all = rebuild_qa_all(cfg, log)

    summary = summarise(entries, counts, opts, time.monotonic() - t_start, n_all)
    summary["interrupted"] = interrupted
    summary["batch"] = batch
    c = summary["counts"]
    log.info("summary: ok=%d parse_error=%d error=%d dry_run=%d | tokens prompt=%d completion=%d reasoning=%d | "
             "est_cost=%s | mean_latency=%s s | elapsed=%.0f s",
             c["ok"], c["parse_error"], c["error"], c["dry_run"], summary["tokens"]["prompt"],
             summary["tokens"]["completion"], summary["tokens"]["reasoning"],
             summary["est_cost_usd"] if summary["est_cost_usd"] is not None else "n/a",
             summary["mean_latency_s"] if summary["mean_latency_s"] is not None else "n/a", summary["elapsed_s"])
    record_run(cfg, STAGE, args, summary, started_at=started)
    if interrupted:
        return 1
    attempted = c["ok"] + c["parse_error"] + c["error"]
    if not dry_run and attempted and c["ok"] == 0:
        log.error("every attempted sample failed")
        return 1
    if dry_run and entries and all(e.status == "error" for e in entries):
        return 1
    return 0


def _run_pool(items: list[WorkItem], client: GatewayClient, cfg: Config, opts: Options,
              log: logging.Logger) -> tuple[list[GenerationLogEntry], bool]:
    """Thread pool over samples; appends to generation_log.jsonl as results arrive."""
    logw = JsonlWriter(Path(cfg.paths.qa) / "generation_log.jsonl")
    entries: list[GenerationLogEntry] = []
    done_lock = threading.Lock()
    interrupted = False
    ex = ThreadPoolExecutor(max_workers=opts.concurrency, thread_name_prefix="gen")
    futures = {ex.submit(generate_one, it, client, cfg, opts, log): it for it in items}
    try:
        for i, fut in enumerate(as_completed(futures), start=1):
            e = fut.result()
            logw.write(e.to_dict())
            with done_lock:
                entries.append(e)
            tok = f"tokens={e.prompt_tokens}/{e.completion_tokens}/{e.reasoning_tokens}" if e.prompt_tokens else ""
            msg = f"[{i}/{len(items)}] {e.sample_id}: {e.status} {e.latency_s or 0:.0f}s {tok} attempts={e.attempts}"
            if e.error:
                msg += f" - {e.error}"
            (log.info if e.status == "ok" else log.warning)("%s", msg)
    except KeyboardInterrupt:
        interrupted = True
        pending = sum(1 for f in futures if not f.done())
        log.warning("interrupted: cancelling %d pending sample(s); in-flight requests finish in the background", pending)
        ex.shutdown(wait=False, cancel_futures=True)
    else:
        ex.shutdown(wait=True)
    return entries, interrupted


if __name__ == "__main__":  # standalone: python -m bench.generate [options]
    import os
    import sys

    from bench.config import load_config

    _p = argparse.ArgumentParser(description="ophbench generate stage")
    add_args(_p)
    _p.add_argument("--config", default=None)
    _p.add_argument("--data-dir", default=None)
    _a = _p.parse_args()
    if _a.data_dir:
        os.environ["OPHBENCH_DATA_DIR"] = _a.data_dir
    sys.exit(main(_a, load_config(_a.config)))
