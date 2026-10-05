"""Adapter registry. Modules are imported lazily so a missing adapter does not break the others."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from bench.schema import DATASETS

if TYPE_CHECKING:  # pragma: no cover
    from bench.datasets.base import DatasetAdapter

# name -> "module:ClassName" (all modules live in bench.datasets)
ADAPTER_SPECS: dict[str, str] = {
    "cataract101": "cataract101:Cataract101Adapter",
    "cataract1k": "cataract1k:Cataract1kAdapter",
    "lmm_phase": "cataract_lmm:LmmPhaseAdapter",
    "lmm_skill": "cataract_lmm:LmmSkillAdapter",
    "lmm_raw": "cataract_lmm:LmmRawAdapter",
    "migs": "migs:MigsAdapter",
    "ophnet": "ophnet:OphNetAdapter",
    "ophora": "ophora:OphoraAdapter",
}

assert set(ADAPTER_SPECS) == set(DATASETS)


def get_adapter_class(name: str):
    if name not in ADAPTER_SPECS:
        raise KeyError(f"unknown dataset '{name}'; known: {sorted(ADAPTER_SPECS)}")
    mod_name, cls_name = ADAPTER_SPECS[name].split(":")
    mod = importlib.import_module(f"bench.datasets.{mod_name}")
    return getattr(mod, cls_name)


def get_adapter(name: str, cfg, log=None) -> "DatasetAdapter":
    return get_adapter_class(name)(cfg, log=log)


def available_adapters() -> list[str]:
    """Names whose module imports successfully (useful while adapters are being written)."""
    out = []
    for n in ADAPTER_SPECS:
        try:
            get_adapter_class(n)
            out.append(n)
        except Exception:
            pass
    return out
