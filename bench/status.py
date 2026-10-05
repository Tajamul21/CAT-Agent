"""Status stage (DESIGN.md §10): one table summarising the progress of every pipeline stage.

Per dataset: inventory records, sampled, prepared ok/err (``data/prepared/_status.jsonl``, latest
entry per sample), generated ok/parse_error/error (``data/qa/generation_log.jsonl``, latest entry
per sample), validated ok/with-issues (``data/qa/qa_validated.jsonl``) and exported
(``ui/data/index.json``). Then token totals / estimated cost over *all* generation calls and the
last run of each stage from ``logs/runs.jsonl``. Read-only apart from its own log file.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from rich import box
from rich.console import Console
from rich.table import Table

from bench.batches import batch_size, n_batches, order_samples, select_batch
from bench.config import Config
from bench.export_ui import ui_paths
from bench.log import get_logger, now_iso, record_run
from bench.schema import DATASETS
from bench.util import iter_jsonl, read_json

STAGE = "status"
GEN_STATUSES = ("ok", "parse_error", "error", "dry_run")


# ------------------------------------------------------------------------------------ helpers
def dataset_of(sample_id: str) -> str:
    ds = (sample_id or "").split("__", 1)[0]
    return ds if ds in DATASETS else "other"


def count_lines(path: Path) -> Optional[int]:
    """Number of non-empty lines, or None when the file is missing."""
    if not path.exists():
        return None
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def latest_by_sample(path: Path) -> dict[str, dict]:
    """Last JSONL entry per sample_id (file order == chronological order)."""
    latest: dict[str, dict] = {}
    for row in iter_jsonl(path):
        sid = row.get("sample_id")
        if sid:
            latest[sid] = row
    return latest


def _fmt_int(v: Optional[int]) -> str:
    return "-" if v is None else f"{v:,}"


def _fmt_pair(a: Optional[int], b: Optional[int], present: bool) -> str:
    return f"{a or 0}/{b or 0}" if present else "-"


# ------------------------------------------------------------------------------------ collection
def collect_status(cfg: Config) -> dict[str, Any]:
    """Gather every count the table needs; tolerant to missing files."""
    data = Path(cfg.paths.data)
    per: dict[str, dict] = {ds: defaultdict(int) for ds in DATASETS}
    per["other"] = defaultdict(int)
    present = {"inventory": False, "sample": False, "prepared": False, "generated": False, "validated": False,
               "exported": False}

    for ds in DATASETS:
        n = count_lines(Path(cfg.paths.inventory) / f"{ds}.jsonl")
        per[ds]["inventory"] = n if n is not None else None  # type: ignore[assignment]
        present["inventory"] |= n is not None

    manifest = Path(cfg.paths.sample) / "sample_manifest.jsonl"
    manifest_rows: list[dict] = []
    ids: dict[str, set[str]] = {k: set() for k in ("prepared_ok", "gen_ok", "val_ok", "exported")}
    if manifest.exists():
        present["sample"] = True
        for row in iter_jsonl(manifest):
            per[dataset_of(row.get("sample_id", ""))]["sampled"] += 1
            manifest_rows.append({"sample_id": row.get("sample_id", ""), "dataset": row.get("dataset", "")})

    status_path = Path(cfg.paths.prepared) / "_status.jsonl"
    if status_path.exists():
        present["prepared"] = True
        for sid, row in latest_by_sample(status_path).items():
            per[dataset_of(sid)]["prepared_ok" if row.get("ok") else "prepared_err"] += 1
            if row.get("ok"):
                ids["prepared_ok"].add(sid)

    gen_log = Path(cfg.paths.qa) / "generation_log.jsonl"
    tokens = Counter()
    n_calls = 0
    cost_known = False
    if gen_log.exists():
        present["generated"] = True
        latest: dict[str, dict] = {}
        for row in iter_jsonl(gen_log):
            n_calls += 1
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                if isinstance(row.get(key), (int, float)):
                    tokens[key] += int(row[key])
            if isinstance(row.get("est_cost_usd"), (int, float)):
                tokens["est_cost_usd_x1e6"] += int(round(float(row["est_cost_usd"]) * 1e6))
                cost_known = True
            if row.get("status") == "dry_run":
                tokens["dry_runs"] += 1
            if row.get("sample_id"):
                latest[row["sample_id"]] = row
        for sid, row in latest.items():
            st = str(row.get("status") or "error")
            per[dataset_of(sid)][f"gen_{st if st in GEN_STATUSES else 'error'}"] += 1
            if st == "ok":
                ids["gen_ok"].add(sid)

    validated = Path(cfg.paths.qa) / "qa_validated.jsonl"
    if validated.exists():
        present["validated"] = True
        for row in iter_jsonl(validated):
            ok = bool((row.get("validation") or {}).get("ok"))
            per[dataset_of(row.get("sample_id", ""))]["val_ok" if ok else "val_issues"] += 1
            if ok:
                ids["val_ok"].add(str(row.get("sample_id", "")))

    ui = ui_paths(cfg)
    index = ui.data / "index.json"
    if index.exists():
        try:
            entries = read_json(index)
            present["exported"] = True
            for e in entries:
                per[dataset_of(e.get("sample_id", ""))]["exported"] += 1
                ids["exported"].add(str(e.get("sample_id", "")))
        except Exception:
            pass

    size = batch_size(cfg)
    batches: list[dict] = []
    if manifest_rows:
        ordered = order_samples(manifest_rows)
        for b in range(1, n_batches(len(ordered), size) + 1):
            members = select_batch(ordered, b, size)
            member_ids = {m["sample_id"] for m in members}
            ds_counts = Counter(m["dataset"] for m in members)
            batches.append({
                "batch": b, "n": len(members),
                "prepared": len(member_ids & ids["prepared_ok"]), "generated": len(member_ids & ids["gen_ok"]),
                "validated": len(member_ids & ids["val_ok"]), "exported": len(member_ids & ids["exported"]),
                "datasets": dict(sorted(ds_counts.items())),
            })

    last_runs: dict[str, dict] = {}
    for row in iter_jsonl(Path(cfg.paths.logs) / "runs.jsonl"):
        if row.get("stage"):
            last_runs[row["stage"]] = row

    return {
        "per_dataset": per,
        "present": present,
        "batches": batches,
        "batch_size": size,
        "tokens": {
            "prompt": tokens["prompt_tokens"], "completion": tokens["completion_tokens"],
            "reasoning": tokens["reasoning_tokens"], "n_calls": n_calls, "dry_runs": tokens["dry_runs"],
            "est_cost_usd": round(tokens["est_cost_usd_x1e6"] / 1e6, 4) if cost_known else None,
        },
        "last_runs": last_runs,
        "paths": {"data": str(data), "logs": str(cfg.paths.logs), "ui": str(ui.root)},
    }


def totals(per: dict[str, dict], key: str) -> Optional[int]:
    vals = [d.get(key) for d in per.values()]
    vals = [v for v in vals if isinstance(v, int)]
    return sum(vals) if vals else None


def next_step_hint(status: dict[str, Any]) -> str:
    p, per = status["present"], status["per_dataset"]
    if not p["inventory"]:
        return "next: ./ophbench inventory"
    if not p["sample"]:
        return "next: ./ophbench sample"
    n_sampled = totals(per, "sampled") or 0
    if (totals(per, "prepared_ok") or 0) < n_sampled:
        return "next: ./ophbench prepare   (resumable)"
    if (totals(per, "gen_ok") or 0) < n_sampled:
        return "next: ./ophbench generate  (resumable; --dry-run to preview prompts)"
    if not p["validated"]:
        return "next: ./ophbench validate"
    if not p["exported"]:
        return "next: ./ophbench export-ui"
    return "all stages have output; re-run stages after changes (status is read-only)"


# ------------------------------------------------------------------------------------ rendering
def render(status: dict[str, Any], console: Console) -> None:
    per, present = status["per_dataset"], status["present"]
    table = Table(title=f"ophbench status  -  data: {status['paths']['data']}", box=box.SIMPLE_HEAVY, show_footer=False)
    for col, justify in (("dataset", "left"), ("inventory", "right"), ("sampled", "right"),
                         ("prepared ok/err", "right"), ("generated ok/parse/err", "right"),
                         ("validated ok/issues", "right"), ("exported", "right")):
        table.add_column(col, justify=justify)

    def row_cells(d: dict, name: str) -> list[str]:
        inv = d.get("inventory")
        gen = (f"{d.get('gen_ok', 0)}/{d.get('gen_parse_error', 0)}/{d.get('gen_error', 0)}"
               + (f" (+{d['gen_dry_run']} dry)" if d.get("gen_dry_run") else "")) if present["generated"] else "-"
        return [
            name,
            _fmt_int(inv) if present["inventory"] else "-",
            _fmt_int(d.get("sampled", 0)) if present["sample"] else "-",
            _fmt_pair(d.get("prepared_ok"), d.get("prepared_err"), present["prepared"]),
            gen,
            _fmt_pair(d.get("val_ok"), d.get("val_issues"), present["validated"]),
            _fmt_int(d.get("exported", 0)) if present["exported"] else "-",
        ]

    for ds in DATASETS:
        table.add_row(*row_cells(per[ds], ds))
    if any(per["other"].values()):
        table.add_row(*row_cells(per["other"], "other"))
    tot = {k: totals(per, k) for k in ("inventory", "sampled", "prepared_ok", "prepared_err", "gen_ok", "gen_parse_error",
                                      "gen_error", "gen_dry_run", "val_ok", "val_issues", "exported")}
    table.add_row(*[f"[bold]{c}[/bold]" for c in row_cells({k: (v or 0) if k != "inventory" else v for k, v in tot.items()}, "TOTAL")])
    console.print(table)

    if status.get("batches"):
        tb = Table(title=f"progress per annotation batch (size {status['batch_size']}; canonical benchmark order)",
                   box=box.SIMPLE, show_header=True)
        for col, justify in (("batch", "right"), ("samples", "right"), ("prepared", "right"), ("generated ok", "right"),
                             ("validated ok", "right"), ("exported", "right"), ("datasets", "left")):
            tb.add_column(col, justify=justify)
        shown = status["batches"][:40]
        for b in shown:
            ds = ", ".join(f"{k} {v}" for k, v in b["datasets"].items())
            tb.add_row(str(b["batch"]), str(b["n"]), str(b["prepared"]), str(b["generated"]), str(b["validated"]),
                       str(b["exported"]), ds)
        if len(status["batches"]) > len(shown):
            tb.add_row("...", f"(+{len(status['batches']) - len(shown)} more batches)", "", "", "", "", "")
        console.print(tb)

    tk = status["tokens"]
    cost = f"${tk['est_cost_usd']:,.2f}" if tk["est_cost_usd"] is not None else "n/a (set llm.price_per_m_*_usd in config)"
    t2 = Table(title="generation tokens / cost (all calls incl. retries)", box=box.SIMPLE, show_header=True)
    for col in ("calls", "dry runs", "prompt tokens", "completion tokens", "reasoning tokens", "est. cost"):
        t2.add_column(col, justify="right")
    t2.add_row(f"{tk['n_calls']:,}", f"{tk['dry_runs']:,}", f"{tk['prompt']:,}", f"{tk['completion']:,}", f"{tk['reasoning']:,}", cost)
    console.print(t2)

    if status["last_runs"]:
        t3 = Table(title="last run per stage (logs/runs.jsonl)", box=box.SIMPLE)
        t3.add_column("stage")
        t3.add_column("finished")
        t3.add_column("summary")
        for stage, row in sorted(status["last_runs"].items()):
            summ = row.get("summary") or {}
            brief = ", ".join(f"{k}={v}" for k, v in summ.items() if not isinstance(v, (dict, list)))[:160]
            t3.add_row(stage, str(row.get("finished_at") or row.get("ts") or ""), brief)
        console.print(t3)
    console.print(f"[dim]logs: {status['paths']['logs']}   ui: {status['paths']['ui']}[/dim]")
    console.print(next_step_hint(status))


# ------------------------------------------------------------------------------------ stage entry
def add_args(p: argparse.ArgumentParser) -> None:
    """``status`` takes no stage-specific options."""


def main(args: argparse.Namespace, cfg: Config) -> int:
    cfg.ensure_dirs()
    log = get_logger(STAGE, cfg)
    started = now_iso()
    status = collect_status(cfg)
    render(status, Console(width=None if sys.stdout.isatty() else 150))
    per = status["per_dataset"]
    summary = {k: totals(per, k) for k in ("inventory", "sampled", "prepared_ok", "prepared_err", "gen_ok", "gen_parse_error",
                                          "gen_error", "val_ok", "val_issues", "exported")}
    summary["tokens"] = status["tokens"]
    summary["n_batches"] = len(status.get("batches") or [])
    summary["batches_complete"] = sum(1 for b in status.get("batches") or [] if b["exported"] >= b["n"] > 0)
    log.debug("status summary: %s", summary)
    record_run(cfg, STAGE, args, summary, started_at=started)
    return 0
