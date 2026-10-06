"""Usage ledger: every GPT call is recorded with who made it, what it was for, tokens and estimated cost.

Each HTTP call to the gateway (generation, retries, JSON repairs, probes; successful or failed) appends
one JSON line to ``<data_dir>/usage/usage_ledger.jsonl``::

    ts, user, os_user, host, key_id, stage, purpose, sample_id, dataset, batch, model, route, effort,
    status (ok|error), http_status, error, prompt_tokens, cached_tokens, completion_tokens,
    reasoning_tokens, total_tokens, latency_s, request_id, attempt, est_cost_usd, prices, source

* ``user`` is ``--user`` / ``$OPHBENCH_USER`` when given, else the login name; ``key_id`` is a short hash
  of the API key (never the key itself), so usage can also be split per key.
* Costs are computed from the token counts with the prices in ``config/pipeline.yaml: llm.price_*``
  (defaults: public GPT-6 Astra list prices; the JHU gateway invoice is authoritative). Reports always
  recompute cost from tokens with the current prices, so fixing a price later fixes the history too.
* ``./ophbench usage`` prints totals per user, day, batch, dataset, stage and model and writes
  ``usage_report.md``, ``usage_summary.json`` and ``usage_calls.csv`` next to the ledger.
  ``--backfill`` imports calls recorded only in ``qa/generation_log.jsonl`` (runs before this ledger).
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import socket
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from bench.config import Config
from bench.log import JsonlWriter, get_logger, now_iso, record_run
from bench.util import iter_jsonl, md_table, write_csv, write_json

LEDGER_NAME = "usage_ledger.jsonl"
USER_ENV = "OPHBENCH_USER"
TOKEN_KEYS = ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
CALL_FIELDS = ["ts", "user", "os_user", "host", "key_id", "stage", "purpose", "sample_id", "dataset", "batch",
               "model", "route", "effort", "status", "http_status", "error", "prompt_tokens", "cached_tokens",
               "completion_tokens", "reasoning_tokens", "total_tokens", "latency_s", "request_id", "attempt",
               "est_cost_usd", "source"]


# ------------------------------------------------------------------------------------ identity
def os_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - no login name in some containers
        return os.environ.get("USER") or "unknown"


def current_user() -> str:
    return (os.environ.get(USER_ENV) or "").strip() or os_user()


def key_id(api_key: Optional[str]) -> Optional[str]:
    """Short, non-reversible fingerprint of the API key (first 10 hex chars of its SHA-256)."""
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:10]


def usage_dir(cfg: Config) -> Path:
    return Path(cfg.paths.data) / "usage"


def ledger_path(cfg: Config) -> Path:
    return usage_dir(cfg) / LEDGER_NAME


# ------------------------------------------------------------------------------------ pricing
@dataclass
class Pricing:
    """USD per million tokens. Reasoning tokens are billed as output (they are part of completion_tokens)."""

    input: Optional[float] = None
    cached_input: Optional[float] = None
    output: Optional[float] = None
    long_context_threshold: Optional[int] = None
    input_long: Optional[float] = None
    output_long: Optional[float] = None

    @classmethod
    def from_cfg(cls, cfg: Config) -> "Pricing":
        llm = cfg.section("llm")

        def f(key: str) -> Optional[float]:
            v = llm.get(key)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        thr = llm.get("price_long_context_threshold_tokens")
        return cls(input=f("price_per_m_input_usd"), cached_input=f("price_per_m_cached_input_usd"),
                   output=f("price_per_m_output_usd"),
                   long_context_threshold=int(thr) if thr else None,
                   input_long=f("price_per_m_input_long_usd"), output_long=f("price_per_m_output_long_usd"))

    @property
    def priced(self) -> bool:
        return self.input is not None and self.output is not None

    def describe(self) -> str:
        if not self.priced:
            return "no prices configured (llm.price_per_m_input_usd / price_per_m_output_usd)"
        s = f"${self.input:g} input, ${self.output:g} output"
        if self.cached_input is not None:
            s += f", ${self.cached_input:g} cached input"
        s += " per million tokens"
        if self.long_context_threshold and self.input_long is not None and self.output_long is not None:
            s += f" (prompts over {self.long_context_threshold:,} tokens: ${self.input_long:g} / ${self.output_long:g})"
        return s

    def cost(self, prompt: Optional[int], cached: Optional[int], completion: Optional[int]) -> Optional[float]:
        if not self.priced or (prompt is None and completion is None):
            return None
        p, c, k = int(prompt or 0), int(completion or 0), int(cached or 0)
        k = min(k, p)
        pin, pout = self.input, self.output
        if self.long_context_threshold and p > self.long_context_threshold:
            pin = self.input_long if self.input_long is not None else pin
            pout = self.output_long if self.output_long is not None else pout
        pcache = self.cached_input if self.cached_input is not None else pin
        return round(((p - k) * pin + k * pcache + c * pout) / 1e6, 6)


# ------------------------------------------------------------------------------------ ledger
class Ledger:
    """Append-only, thread-safe usage ledger. Recording never raises (usage must not break a run)."""

    def __init__(self, cfg: Config, log: Any = None):
        self.cfg = cfg
        self.path = ledger_path(cfg)
        self.pricing = Pricing.from_cfg(cfg)
        self.log = log
        self._writer: Optional[JsonlWriter] = None

    def record(self, **fields: Any) -> Optional[dict]:
        try:
            entry = {k: None for k in CALL_FIELDS}
            entry.update(ts=now_iso(), user=current_user(), os_user=os_user(), host=socket.gethostname(),
                         status="ok", source="live")
            entry.update({k: v for k, v in fields.items() if k in CALL_FIELDS or k == "prices"})
            entry["est_cost_usd"] = self.pricing.cost(entry.get("prompt_tokens"), entry.get("cached_tokens"),
                                                      entry.get("completion_tokens"))
            entry["prices"] = self.pricing.describe() if self.pricing.priced else None
            if self._writer is None:
                self._writer = JsonlWriter(self.path)
            self._writer.write(entry)
            return entry
        except Exception as e:  # pragma: no cover - disk full etc.
            if self.log is not None:
                self.log.warning("could not write the usage ledger (%s): %s", self.path, e)
            return None


def read_ledger(cfg: Config) -> list[dict]:
    return list(iter_jsonl(ledger_path(cfg)))


# ------------------------------------------------------------------------------------ backfill
def backfill_from_generation_log(cfg: Config, user: Optional[str] = None) -> int:
    """Import generation-log entries whose request_id (or sample+time) is not in the ledger yet."""
    gen_log = Path(cfg.paths.qa) / "generation_log.jsonl"
    if not gen_log.exists():
        return 0
    rows = read_ledger(cfg)
    seen_rid = {r.get("request_id") for r in rows if r.get("request_id")}
    seen_key = {(r.get("sample_id"), r.get("ts")) for r in rows}
    try:
        from bench.batches import manifest_batch_map
        bmap = manifest_batch_map(cfg)
    except Exception:
        bmap = {}
    ledger = Ledger(cfg)
    n = 0
    for g in iter_jsonl(gen_log):
        rid = g.get("request_id")
        ts = g.get("finished_at") or g.get("started_at")
        if (rid and rid in seen_rid) or (g.get("sample_id"), ts) in seen_key:
            continue
        sid = g.get("sample_id")
        if user:
            os.environ[USER_ENV] = user
        ledger.record(ts=ts, stage="generate", purpose="generate (from generation log)", sample_id=sid,
                      dataset=str(sid).split("__")[0] if sid else None, batch=bmap.get(sid), model=g.get("model"),
                      route=g.get("route"), effort=g.get("reasoning_effort"),
                      status="ok" if g.get("status") in ("ok", "parse_error") else "error", error=g.get("error"),
                      prompt_tokens=g.get("prompt_tokens"), completion_tokens=g.get("completion_tokens"),
                      reasoning_tokens=g.get("reasoning_tokens"),
                      total_tokens=(g.get("prompt_tokens") or 0) + (g.get("completion_tokens") or 0)
                      if g.get("prompt_tokens") is not None else None,
                      latency_s=g.get("latency_s"), request_id=rid, attempt=g.get("attempts"), source="backfill")
        n += 1
    return n


# ------------------------------------------------------------------------------------ summaries
def _add(acc: dict, r: dict, cost: Optional[float]) -> None:
    acc["calls"] += 1
    acc["ok"] += r.get("status") == "ok"
    acc["errors"] += r.get("status") != "ok"
    for k in TOKEN_KEYS:
        acc[k] += int(r.get(k) or 0)
    acc["latency_s"] += float(r.get("latency_s") or 0)
    if cost is not None:
        acc["cost_usd"] += cost
    if r.get("sample_id") and r.get("stage") == "generate" and r.get("status") == "ok":
        acc["_videos"].add(r["sample_id"])


def _new() -> dict:
    return {"calls": 0, "ok": 0, "errors": 0, **{k: 0 for k in TOKEN_KEYS}, "latency_s": 0.0, "cost_usd": 0.0,
            "_videos": set()}


def _finish(acc: dict) -> dict:
    out = {k: v for k, v in acc.items() if k != "_videos"}
    out["videos"] = len(acc["_videos"])
    out["cost_usd"] = round(out["cost_usd"], 4)
    out["latency_s"] = round(out["latency_s"], 1)
    out["cost_per_video_usd"] = round(out["cost_usd"] / out["videos"], 4) if out["videos"] else None
    return out


def summarize(rows: Iterable[dict], pricing: Pricing) -> dict:
    groups = {"user": "user", "day": None, "batch": "batch", "dataset": "dataset", "stage": "stage",
              "model": "model", "key": "key_id"}
    total = _new()
    by: dict[str, dict[str, dict]] = {g: defaultdict(_new) for g in groups}
    first = last = None
    n = 0
    for r in rows:
        n += 1
        cost = pricing.cost(r.get("prompt_tokens"), r.get("cached_tokens"), r.get("completion_tokens"))
        _add(total, r, cost)
        ts = str(r.get("ts") or "")
        first = ts if first is None or (ts and ts < first) else first
        last = ts if last is None or ts > last else last
        for g, field in groups.items():
            key = ts[:10] if g == "day" else r.get(field)
            _add(by[g][str(key if key not in (None, "") else "-")], r, cost)
    return {
        "generated_at": now_iso(), "n_calls": n, "first_call": first, "last_call": last,
        "pricing": pricing.describe(), "priced": pricing.priced, "total": _finish(total),
        "by": {g: {k: _finish(v) for k, v in sorted(d.items(), key=lambda kv: kv[0])} for g, d in by.items()},
    }


def _money(v: Any, priced: bool) -> str:
    return f"${v:,.2f}" if priced and v is not None else "-"


def _rows(table: dict, priced: bool) -> list[list[Any]]:
    return [[k, v["calls"], v["errors"], v["videos"], f"{v['prompt_tokens']:,}", f"{v['cached_tokens']:,}",
             f"{v['completion_tokens']:,}", f"{v['reasoning_tokens']:,}", f"{v['total_tokens']:,}",
             _money(v["cost_usd"], priced), _money(v["cost_per_video_usd"], priced)] for k, v in table.items()]


HEADERS = ["", "calls", "errors", "videos", "input tok", "cached tok", "output tok", "reasoning tok", "total tok",
           "est. cost", "cost / video"]


def report_markdown(s: dict) -> str:
    t, priced = s["total"], s["priced"]
    lines = [
        "# GPT usage report", "",
        f"- generated: {s['generated_at']}",
        f"- calls recorded: {s['n_calls']} (from {s['first_call'] or '-'} to {s['last_call'] or '-'})",
        f"- prices used: {s['pricing']}",
        f"- total: **{t['total_tokens']:,} tokens**, estimated **{_money(t['cost_usd'], priced)}** for "
        f"{t['videos']} generated videos ({_money(t['cost_per_video_usd'], priced)} per video)",
        "- estimates use list prices; the WSE AI Gateway dashboard / JHU invoice is the authoritative bill.", "",
    ]
    titles = {"user": "Per person", "day": "Per day", "batch": "Per batch", "dataset": "Per dataset",
              "stage": "Per stage", "model": "Per model", "key": "Per API key (fingerprint)"}
    for g, title in titles.items():
        lines += [f"## {title}", "", md_table([g] + HEADERS[1:], _rows(s["by"][g], priced)), ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ CLI
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backfill", action="store_true",
                   help="first import calls recorded only in qa/generation_log.jsonl (runs before the ledger existed)")
    p.add_argument("--backfill-user", default=None, help="user name to attribute backfilled calls to (default: you)")
    p.add_argument("--person", dest="only_user", default=None, help="only this person (as recorded in the ledger)")
    p.add_argument("--since", default=None, help="only calls on/after this date (YYYY-MM-DD)")


def main(args: argparse.Namespace, cfg: Config) -> int:
    log = get_logger("usage", cfg)
    cfg.ensure_dirs()
    started = now_iso()
    if getattr(args, "backfill", False):
        n = backfill_from_generation_log(cfg, user=getattr(args, "backfill_user", None))
        log.info("backfilled %d call(s) from the generation log", n)
    rows = read_ledger(cfg)
    if getattr(args, "only_user", None):
        rows = [r for r in rows if str(r.get("user", "")).lower() == args.only_user.lower()]
    if getattr(args, "since", None):
        rows = [r for r in rows if str(r.get("ts", ""))[:10] >= args.since]
    pricing = Pricing.from_cfg(cfg)
    s = summarize(rows, pricing)
    out = usage_dir(cfg)
    write_json(out / "usage_summary.json", s)
    (out / "usage_report.md").write_text(report_markdown(s), encoding="utf-8")
    calls = [{k: r.get(k) for k in CALL_FIELDS} for r in rows]
    for c in calls:
        c["est_cost_usd"] = pricing.cost(c.get("prompt_tokens"), c.get("cached_tokens"), c.get("completion_tokens"))
    write_csv(out / "usage_calls.csv", calls, CALL_FIELDS)
    _print(s)
    log.info("usage report: %s (also usage_summary.json, usage_calls.csv)", out / "usage_report.md")
    record_run(cfg, "usage", args, {"n_calls": s["n_calls"], "total_tokens": s["total"]["total_tokens"],
                                     "est_cost_usd": s["total"]["cost_usd"]}, started_at=started)
    return 0


def _print(s: dict) -> None:
    """Compact console view (the markdown report has every column)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:  # pragma: no cover
        print(report_markdown(s))
        return
    con = Console()
    t, priced = s["total"], s["priced"]
    con.print(f"[bold]GPT usage[/bold]  {s['n_calls']} calls · {t['total_tokens']:,} tokens · "
              f"est. {_money(t['cost_usd'], priced)} · {t['videos']} videos ({_money(t['cost_per_video_usd'], priced)}/video)")
    con.print(f"prices: {s['pricing']}")
    for g, title in (("user", "per person"), ("day", "per day"), ("batch", "per batch"), ("dataset", "per dataset")):
        tb = Table(title=title, title_justify="left", pad_edge=False)
        for h in (g, "calls", "err", "videos", "tokens", "est. cost", "$/video"):
            tb.add_column(h, justify="left" if h == g else "right", no_wrap=True)
        for k, v in s["by"][g].items():
            tb.add_row(str(k), str(v["calls"]), str(v["errors"]), str(v["videos"]), f"{v['total_tokens'] / 1e6:.2f}M",
                       _money(v["cost_usd"], priced), _money(v["cost_per_video_usd"], priced))
        con.print(tb)
