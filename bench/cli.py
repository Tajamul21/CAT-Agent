"""ophbench command-line interface (DESIGN.md §§4-10).

Stages, in pipeline order:
  inventory   scan dataset metadata                 -> data/inventory/<dataset>.jsonl (+ summary.json/.md)
  sample      stratified, seeded sampling (1.5K)    -> data/sample/sample_manifest.jsonl (+ report)
  prepare     materialise video, probe, frames, contact sheet, preview -> data/prepared/batch_<KKK>/<sample_id>/
  generate    GPT-6 Astra question generation       -> data/qa/<sample_id>.json, data/qa/generation_log.jsonl
  validate    rule checks on the QA sets            -> data/qa/qa_validated.jsonl, validation_report.json/.md
  export-ui   annotation-UI bundle                  -> ui/data/*.json, ui/media/<sample_id>/
Utilities:
  status      progress table across all stages
  probe       LLM gateway connectivity / model probe (bench.llm_client)
  run-all     inventory -> sample -> prepare -> generate -> validate -> export-ui (stops at first failure)

Global options (accepted before or after the stage name):
  --config PATH     alternative pipeline.yaml       --data-dir PATH   data directory (sets OPHBENCH_DATA_DIR)
  -v / --verbose    DEBUG logging

Usage:  ./ophbench <stage> [options]        or        python3 -m bench.cli <stage> [options]
        ./ophbench <stage> --help           shows the stage's options

Each stage module exposes ``add_args(parser)`` and ``main(args, cfg) -> int``; modules are imported
lazily so a stage that is not implemented yet only breaks its own command.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence

from bench.config import REPO_ROOT, Config, load_config

# stage name -> (module path, one-line help); run-all is handled in this module
STAGES: dict[str, tuple[Optional[str], str]] = {
    "inventory": ("bench.inventory", "scan dataset metadata -> data/inventory/<dataset>.jsonl"),
    "sample": ("bench.sampler", "stratified, seeded sampling -> data/sample/sample_manifest.jsonl"),
    "prepare": ("bench.prepare", "materialise video, probe, frames, contact sheet, preview -> data/prepared/batch_<KKK>/<id>/"),
    "generate": ("bench.generate", "GPT-6 Astra question generation -> data/qa/<id>.json"),
    "validate": ("bench.validate", "rule checks -> data/qa/qa_validated.jsonl + validation_report.*"),
    "export-ui": ("bench.export_ui", "annotation-UI bundle -> ui/data/*.json, ui/media/<id>/"),
    "status": ("bench.status", "progress table across stages"),
    "probe": ("bench.llm_client", "LLM gateway connectivity / model probe"),
    "usage": ("bench.usage", "GPT usage per person/day/batch/dataset: tokens and estimated USD"),
    "run-all": (None, "inventory -> sample -> prepare -> generate -> validate -> export-ui"),
}
RUN_ALL_ORDER: tuple[str, ...] = ("inventory", "sample", "prepare", "generate", "validate", "export-ui")


class StageUnavailable(RuntimeError):
    """Raised when a stage module cannot be imported."""


# ------------------------------------------------------------------------------------ parsers
def _globals_parent() -> argparse.ArgumentParser:
    """Parent parser with the global options (fresh instance each time; argparse copies actions)."""
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--config", metavar="PATH", default=None, help="pipeline YAML (default: config/pipeline.yaml)")
    g.add_argument("--data-dir", metavar="PATH", default=None,
                   help="data directory (default: data/; exported as OPHBENCH_DATA_DIR before config load)")
    g.add_argument("--user", metavar="NAME", default=None,
                   help="who is running this (recorded in the usage ledger; default: $OPHBENCH_USER or login name)")
    g.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    return g


def build_parser() -> argparse.ArgumentParser:
    """Top-level parser: globals + stage names. Stage options are parsed in a second pass."""
    top = argparse.ArgumentParser(
        prog="ophbench",
        description="ophbench - Stage-1 curation pipeline for an agentic ophthalmic-surgery video benchmark.",
        epilog="Run `ophbench <stage> --help` for stage options.",
        parents=[_globals_parent()],
    )
    sub = top.add_subparsers(dest="command", metavar="<stage>")
    for name, (_mod, help_text) in STAGES.items():
        sub.add_parser(name, help=help_text, add_help=False)
    return top


def add_run_all_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--limit", type=int, default=None, help="passed to prepare and generate (--limit N)")
    p.add_argument("--no-preview", action="store_true", help="passed to prepare (skip preview.mp4)")
    p.add_argument("--concurrency", type=int, default=None, help="passed to generate (--concurrency N)")
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh"), default=None,
                   help="passed to generate (--effort E)")


def import_stage(name: str) -> ModuleType:
    """Import the module behind a stage, raising StageUnavailable with a readable message."""
    mod_name = STAGES[name][0]
    if mod_name is None:
        raise StageUnavailable(f"stage '{name}' has no module")
    try:
        return importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        if e.name and (e.name == mod_name or mod_name.startswith(e.name + ".")):
            raise StageUnavailable(
                f"stage '{name}' is not available: module {mod_name} not found (not implemented yet?)") from e
        raise StageUnavailable(f"stage '{name}' cannot be imported: missing dependency {e.name!r} ({e})") from e
    except Exception as e:  # syntax errors, import-time failures
        raise StageUnavailable(f"stage '{name}' failed to import ({mod_name}): {type(e).__name__}: {e}") from e


def build_stage_parser(name: str, module: Optional[ModuleType]) -> argparse.ArgumentParser:
    """Full parser for one stage: globals + the module's ``add_args`` (or run-all's options)."""
    if module is None:
        desc = STAGES[name][1]
    else:
        desc = ((module.__doc__ or "").strip().split("\n\n")[0] or STAGES[name][1]).strip()
    sp = argparse.ArgumentParser(prog=f"ophbench {name}", description=desc, parents=[_globals_parent()])
    if module is None:
        add_run_all_args(sp)
    else:
        add_args = getattr(module, "add_args", None)
        if callable(add_args):
            add_args(sp)
    return sp


# ------------------------------------------------------------------------------------ helpers
def _cli_logger() -> logging.Logger:
    from bench.log import get_logger

    return get_logger("cli")


def _enable_verbose() -> None:
    """Make every stage logger DEBUG: wrap bench.log.get_logger before stage modules bind the name."""
    import bench.log as blog

    logging.getLogger().setLevel(logging.DEBUG)
    original = blog.get_logger
    if getattr(original, "_ophbench_verbose", False):
        return

    def get_logger_verbose(stage: str, cfg=None, level: int = logging.DEBUG) -> logging.Logger:
        logger = original(stage, cfg, level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        return logger

    get_logger_verbose._ophbench_verbose = True  # type: ignore[attr-defined]
    blog.get_logger = get_logger_verbose


def _resolve_path(p: str, invoked_from: Path) -> Path:
    """Relative CLI paths are taken from the invoking directory (fallback: repo root)."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    cand = (invoked_from / path).resolve()
    if cand.exists() or not (REPO_ROOT / path).exists():
        return cand
    return (REPO_ROOT / path).resolve()


def load_cfg(args: argparse.Namespace, invoked_from: Path) -> Config:
    """Apply --data-dir (env OPHBENCH_DATA_DIR) and --config, then load the configuration."""
    if getattr(args, "data_dir", None):
        os.environ["OPHBENCH_DATA_DIR"] = str(_resolve_path(args.data_dir, invoked_from))
    if getattr(args, "user", None):
        os.environ["OPHBENCH_USER"] = str(args.user).strip()
    cfg_path = _resolve_path(args.config, invoked_from) if getattr(args, "config", None) else None
    if cfg_path is not None and not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    return load_config(cfg_path)


def call_stage_main(module: ModuleType, args: argparse.Namespace, cfg: Config, log: logging.Logger) -> int:
    """Run ``module.main(args, cfg)``; any exception becomes rc=1 with a traceback in the log."""
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        log.error("module %s has no main(args, cfg)", module.__name__)
        return 1
    try:
        rc = main_fn(args, cfg)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except SystemExit as e:  # modules should return codes, but be tolerant
        code = e.code
        return 0 if code in (None, 0) else (int(code) if isinstance(code, int) else 1)
    except Exception as e:
        log.error("stage %s crashed: %s: %s", module.__name__, type(e).__name__, e)
        log.debug("traceback", exc_info=True)
        if not log.isEnabledFor(logging.DEBUG):
            log.error("re-run with -v for the traceback")
        return 1
    return 0 if rc is None else int(rc)


def _global_namespace(args: argparse.Namespace, command: str) -> argparse.Namespace:
    return argparse.Namespace(command=command, config=getattr(args, "config", None),
                              data_dir=getattr(args, "data_dir", None), verbose=bool(getattr(args, "verbose", False)))


def run_all_plan(args: argparse.Namespace) -> dict[str, list[str]]:
    """Per-stage argv for run-all (only the pass-through options)."""
    limit = ["--limit", str(args.limit)] if getattr(args, "limit", None) is not None else []
    gen = list(limit)
    if getattr(args, "concurrency", None):
        gen += ["--concurrency", str(args.concurrency)]
    if getattr(args, "effort", None):
        gen += ["--effort", str(args.effort)]
    return {
        "inventory": [],
        "sample": [],
        "prepare": limit + (["--no-preview"] if getattr(args, "no_preview", False) else []),
        "generate": gen,
        "validate": [],
        "export-ui": [],
    }


def run_all(args: argparse.Namespace, cfg: Config) -> int:
    """Run the six pipeline stages in order; stop at the first non-zero return code."""
    from bench.log import get_logger, now_iso, record_run

    log = get_logger("run_all", cfg)
    started = now_iso()
    plan = run_all_plan(args)
    stages_summary: dict[str, dict] = {}
    log.info("run-all: %s", " -> ".join(RUN_ALL_ORDER))
    for stage in RUN_ALL_ORDER:
        argv = plan[stage]
        log.info("=== %s %s", stage, " ".join(argv))
        t0 = time.time()
        try:
            module = import_stage(stage)
            stage_args = build_stage_parser(stage, module).parse_args(argv, namespace=_global_namespace(args, stage))
        except StageUnavailable as e:
            log.error("%s", e)
            stages_summary[stage] = {"rc": 1, "error": str(e)}
            record_run(cfg, "run_all", args, {"ok": False, "failed_stage": stage, "stages": stages_summary}, started_at=started)
            return 1
        rc = call_stage_main(module, stage_args, cfg, log)
        elapsed = round(time.time() - t0, 1)
        stages_summary[stage] = {"rc": rc, "elapsed_s": elapsed}
        if rc != 0:
            log.error("=== %s failed (rc=%s) after %ss - stopping", stage, rc, elapsed)
            record_run(cfg, "run_all", args, {"ok": False, "failed_stage": stage, "stages": stages_summary}, started_at=started)
            return rc
        log.info("=== %s ok (%ss)", stage, elapsed)
    record_run(cfg, "run_all", args, {"ok": True, "stages": stages_summary}, started_at=started)
    log.info("run-all finished: all stages ok")
    return 0


# ------------------------------------------------------------------------------------ entry point
def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    invoked_from = Path.cwd()
    os.chdir(REPO_ROOT)
    if any(a in ("-v", "--verbose") for a in argv):
        _enable_verbose()

    top = build_parser()
    ns, rest = top.parse_known_args(argv)
    if not ns.command:
        top.print_help(sys.stderr)
        return 2
    log = _cli_logger()

    module: Optional[ModuleType] = None
    if ns.command != "run-all":
        try:
            module = import_stage(ns.command)
        except StageUnavailable as e:
            log.error("%s", e)
            log.debug("import failure", exc_info=True)
            return 1
    args = build_stage_parser(ns.command, module).parse_args(rest, namespace=ns)

    try:
        cfg = load_cfg(args, invoked_from)
    except Exception as e:
        log.error("could not load configuration: %s", e)
        return 1
    log.debug("config: %s  data: %s  logs: %s", cfg.source, cfg.paths.data, cfg.paths.logs)

    if module is None:
        try:
            return run_all(args, cfg)
        except KeyboardInterrupt:
            log.error("interrupted")
            return 130
    return call_stage_main(module, args, cfg, log)


if __name__ == "__main__":
    sys.exit(main())
