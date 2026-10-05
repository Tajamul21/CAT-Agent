"""Configuration loading for ophbench (DESIGN.md §11).

``load_config()`` reads ``config/pipeline.yaml`` (deep-merged over DEFAULTS), applies environment
overrides (OPHBENCH_DATA_DIR, OPHBENCH_GATEWAY_BASE, OPHBENCH_MODEL, OPHBENCH_EFFORT) and exposes
resolved paths under ``cfg.paths``.
"""
from __future__ import annotations

import copy
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "seed": 20261005,
    "datasets_root": "/mnt/store/tashraf4/datasets",
    "data_dir": "data",
    "logs_dir": "logs",
    "ffmpeg": "auto",
    "ffprobe": "auto",
    "workers": 12,
    "target_total": 1500,
    "quotas": {
        "cataract101": 101, "cataract1k": 300, "lmm_phase": 150, "lmm_skill": 120,
        "lmm_raw": 80, "migs": 160, "ophnet": 420, "ophora": 169,
    },
    "sampling": {
        "duration_bins_s": [300, 480, 720],
        "cataract1k": {"annotated_share": 0.87, "rare_flag_first": True},
        "lmm_skill": {"include_all_adverse": True},
        "lmm_raw": {"min_duration_s": 300, "max_duration_s": 1500, "s2_min": 30, "exclude_phase_subset": True},
        "migs": {"min_per_operation_type": 1},
        "ophnet": {"unit": "case", "max_case_duration_s": 1200, "min_case_duration_s": 60, "min_phases": 3, "max_per_case": 1},
        "ophora": {
            "one_clip_per_source_video": True, "min_instruction_words": 12,
            "category_weights": {
                "cataract": 1.0, "glaucoma": 2.0, "cornea": 2.0, "retina": 1.5,
                "oculoplastics_strabismus": 2.0, "refractive": 1.5, "other_mixed": 1.0,
            },
        },
    },
    "media": {
        "frames": {"seconds_per_frame": 10, "min_frames": 12, "max_frames": 48, "resize_long_side": 768,
                   "jpeg_quality": 85, "boundary_frames": True},
        "contact_sheet": {"cols": 6, "tile_width": 320},
        "preview": {"enabled": True, "height": 360, "crf": 28, "max_duration_s": None},
    },
    "llm": {
        "base_url": "https://gateway.engineering.jhu.edu/gateway",
        "api_key_env": "GPT_ASTRA_KEY",
        "api_key_file": "/mnt/store/tashraf4/.api_keys.env",
        "route": "responses",
        "model": "openai/gpt-6-astra",
        "reasoning_effort": "xhigh",
        "max_completion_tokens": 32000,
        "image_detail": "high",
        "stream": True,
        "concurrency": 4,
        "timeout_s": 1200,
        "max_retries": 5,
        "questions_per_video": 3,
        "prompt_version": "v1",
        "price_per_m_input_usd": None,
        "price_per_m_output_usd": None,
    },
    "batches": {"size": 100, "package_kind": "preview"},
    "ui": {
        "media_base_url": "",
        "submit_url": "",
        "questions_to_select_min": 1,
        "questions_to_select_max": 2,
        "annotators": [],
        "overlap_fraction": 0.1,
    },
}

_KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    def __init__(self, raw: dict, root: Path = REPO_ROOT, source: Optional[Path] = None):
        self.raw = raw
        self.root = Path(root)
        self.source = source
        # environment overrides
        env_base = os.environ.get("OPHBENCH_GATEWAY_BASE")
        if env_base:
            self.raw.setdefault("llm", {})["base_url"] = env_base.rstrip("/")
        env_model = os.environ.get("OPHBENCH_MODEL")
        if env_model:
            self.raw.setdefault("llm", {})["model"] = env_model
        env_effort = os.environ.get("OPHBENCH_EFFORT")
        if env_effort:
            self.raw.setdefault("llm", {})["reasoning_effort"] = env_effort
        data_dir = Path(os.environ.get("OPHBENCH_DATA_DIR") or raw.get("data_dir", "data"))
        if not data_dir.is_absolute():
            data_dir = self.root / data_dir
        logs_dir = Path(os.environ.get("OPHBENCH_LOGS_DIR") or raw.get("logs_dir", "logs"))
        if not logs_dir.is_absolute():
            logs_dir = self.root / logs_dir
        ui_dir = self.root / "ui"
        self.paths = SimpleNamespace(
            root=self.root,
            data=data_dir,
            logs=logs_dir,
            inventory=data_dir / "inventory",
            sample=data_dir / "sample",
            prepared=data_dir / "prepared",
            cache=data_dir / "cache",
            qa=data_dir / "qa",
            annotations=data_dir / "annotations",
            ui=ui_dir,
            ui_data=ui_dir / "data",
            ui_media=ui_dir / "media",
            packages=Path(os.environ.get("OPHBENCH_PACKAGES_DIR") or raw.get("packages_dir") or (self.root / "ui_packages")),
            resources=self.root / "resources",
            config=self.root / "config",
            prompts=self.root / "config" / "prompts",
            datasets_root=Path(raw.get("datasets_root", DEFAULTS["datasets_root"])),
        )

    # ---- accessors -------------------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def section(self, name: str) -> dict:
        v = self.raw.get(name, {})
        return v if isinstance(v, dict) else {}

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", DEFAULTS["seed"]))

    @property
    def workers(self) -> int:
        return int(self.raw.get("workers", DEFAULTS["workers"]))

    def ensure_dirs(self) -> None:
        for p in (self.paths.data, self.paths.logs, self.paths.inventory, self.paths.sample, self.paths.prepared,
                  self.paths.cache, self.paths.qa, self.paths.annotations):
            Path(p).mkdir(parents=True, exist_ok=True)

    # ---- tools -----------------------------------------------------------------------------
    def ffmpeg(self) -> str:
        v = str(self.raw.get("ffmpeg", "auto"))
        if v != "auto":
            return v
        try:
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            found = shutil.which("ffmpeg")
            if not found:
                raise RuntimeError("ffmpeg not found: install imageio-ffmpeg or set `ffmpeg:` in config")
            return found

    def ffprobe(self) -> Optional[str]:
        v = str(self.raw.get("ffprobe", "auto"))
        if v != "auto":
            return v if Path(v).exists() else None
        cand = Path("/mnt/store/tashraf4/miniconda3/envs/opencompass/bin/ffprobe")
        if cand.exists():
            return str(cand)
        return shutil.which("ffprobe")

    # ---- secrets ---------------------------------------------------------------------------
    def api_key(self) -> str:
        llm = self.section("llm")
        env_name = llm.get("api_key_env", "GPT_ASTRA_KEY")
        key = os.environ.get(env_name, "").strip()
        if key:
            return key
        f = Path(llm.get("api_key_file", "")) if llm.get("api_key_file") else None
        if f and f.exists():
            for line in f.read_text().splitlines():
                m = _KEY_RE.match(line)
                if m and m.group(1) == env_name:
                    return m.group(2).strip().strip("'\"")
        raise RuntimeError(f"API key not found: set ${env_name} or add it to {f}")

    def snapshot(self) -> dict:
        """Config as written to logs/reports (the YAML holds no secrets)."""
        snap = copy.deepcopy(self.raw)
        snap["_resolved"] = {
            "root": str(self.root), "data_dir": str(self.paths.data), "logs_dir": str(self.paths.logs),
            "source": str(self.source) if self.source else None,
        }
        return snap


def load_config(path: Optional[str | Path] = None, root: Optional[Path] = None) -> Config:
    root = Path(root) if root else REPO_ROOT
    p = Path(path) if path else root / "config" / "pipeline.yaml"
    raw: dict = {}
    if p.exists():
        with open(p) as fh:
            raw = yaml.safe_load(fh) or {}
    merged = deep_merge(DEFAULTS, raw)
    return Config(merged, root=root, source=p if p.exists() else None)
