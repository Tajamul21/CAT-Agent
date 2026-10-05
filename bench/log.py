"""Logging helpers (DESIGN.md §2): rich console + per-run file log, JSONL event logs, run records."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
except Exception:  # pragma: no cover - rich is installed, but keep a fallback
    Console = None  # type: ignore
    RichHandler = None  # type: ignore

_LOGGERS: dict[str, logging.Logger] = {}
_KEY_RE = re.compile(r"jhu_live_sk_[A-Za-z0-9_\-]+|sk-[A-Za-z0-9_\-]{20,}")
_B64_RE = re.compile(r"data:[a-z]+/[a-z0-9.+\-]+;base64,[A-Za-z0-9+/=]{64,}")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_logger(stage: str, cfg: Any = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger named ophbench.<stage>; adds a file handler under cfg.paths.logs when cfg is given."""
    if stage in _LOGGERS:
        return _LOGGERS[stage]
    logger = logging.getLogger(f"ophbench.{stage}")
    logger.setLevel(level)
    logger.propagate = False
    if RichHandler is not None:
        ch: logging.Handler = RichHandler(console=Console(stderr=True), show_path=False, rich_tracebacks=False,
                                          markup=False, log_time_format="%H:%M:%S")
        ch.setFormatter(logging.Formatter("%(message)s"))
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
    if cfg is not None:
        logs_dir = Path(cfg.paths.logs)
        logs_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{stage}_{_ts()}.log"
        fh = logging.FileHandler(logs_dir / fname, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logger.addHandler(fh)
        link = logs_dir / f"{stage}.latest.log"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(fname)
        except OSError:
            pass
        logger.info("log file: %s", logs_dir / fname)
    _LOGGERS[stage] = logger
    return logger


def redact(obj: Any) -> Any:
    """Recursively replace secrets and base64 payloads in a JSON-like object."""
    if isinstance(obj, str):
        s = _KEY_RE.sub("<redacted-key>", obj)
        return _B64_RE.sub(lambda m: f"<base64 {len(m.group(0))} chars>", s)
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


class JsonlWriter:
    """Append-only JSON lines writer, safe to share between threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, obj: dict) -> None:
        rec = dict(obj)
        rec.setdefault("ts", now_iso())
        line = json.dumps(redact(rec), ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def record_run(cfg: Any, stage: str, args: Any, summary: dict, started_at: Optional[str] = None) -> None:
    """Append a run record to logs/runs.jsonl."""
    if hasattr(args, "__dict__"):
        args = {k: v for k, v in vars(args).items() if not k.startswith("_") and k != "func"}
    JsonlWriter(Path(cfg.paths.logs) / "runs.jsonl").write({
        "stage": stage,
        "started_at": started_at,
        "finished_at": now_iso(),
        "args": args,
        "summary": summary,
        "pid": os.getpid(),
    })
