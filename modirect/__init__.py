"""modirect — direction concept vectors and magnitude interventions for VLMs.

This package WRAPS the original runtime in `core/`; it does not replace or copy it.

Nothing is imported eagerly here, and that is load-bearing. `core/__init__.py:1` pulls
in `core.model_loader`, which imports `llava` at module top level
(`core/model_loader.py:10`). LLaVA-NeXT is not on PyPI and is absent from most hosts,
so an `import core` at this level would make `import modirect` — and therefore the
entire test suite — unimportable on any machine without the model runtime.

A second, quieter reason to import nothing eagerly: `core/dataset_loader.py:119` computes
`_DEFAULT_TASKS_DIR` from `__file__` and runs `discover_tasks()` AT IMPORT TIME. Reaching
it as a side effect of `import modirect` would fire a filesystem scan, and a wrong path
there does not raise — it leaves a SILENTLY EMPTY registry that surfaces much later as a
puzzling "Unknown task".

Import `core` lazily, inside the function that needs it:

    def load_model(...):
        from modirect.runtime import load_core   # imports core at call time, not now

The pure modules (`modirect.concepts`, `modirect.interventions.operators`) depend on
numpy only and are exercised by tests with neither torch nor llava installed.

Accordingly, the names below resolve through a PEP 562 module `__getattr__`: `import
modirect` binds nothing but `__version__`, and each subpackage is imported on first touch.
Even then torch stays out — `probing.linear` and `io.concept_store` defer `import torch`
into the functions that need it, so only an actual probe fit or `.pt` read requires it.

    import modirect                        # numpy only; touches no subpackage
    modirect.extract_concept_vectors(...)  # imports modirect.concepts now
    modirect.train_linear_probe(...)       # imports torch now, at CALL time

Layout:
    config/         paths, model registry, direction orderings, stage/disk layout
    concepts/       Δ_d = mean(h|d) − mean(h) — the project's central quantity
    interventions/  the L20-L21 magnitude operators and the hooks that apply them
    probing/        one linear probe; the direction and letter targets
    io/             cached features on disk; the committed concept vectors
    models/, tasks/ typed spines over core/model_loader.py and core/dataset_loader.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

#: Subpackages, resolved lazily by `__getattr__`.
_SUBMODULES = ("config", "concepts", "interventions", "io", "models", "probing", "tasks")

#: Re-exported name -> the module that defines it. Deliberately a small, curated surface:
#: the things you reach for from a notebook. Everything else stays one dot deeper.
_LAZY: dict[str, str] = {
    # concepts
    "ConceptAxes": "modirect.concepts",
    "extract_concept_vectors": "modirect.concepts",
    "concept_axes_by_layer": "modirect.concepts",
    # interventions
    "apply_condition": "modirect.interventions",
    "project_on_axis": "modirect.interventions",
    "CONDITIONS": "modirect.interventions",
    "last_token_hook": "modirect.interventions",
    "LastTokenIntervention": "modirect.interventions",
    # probing
    "ProbeConfig": "modirect.probing",
    "ProbeResult": "modirect.probing",
    "train_linear_probe": "modirect.probing",
    "get_preset": "modirect.probing",
    "ProbeTarget": "modirect.probing",
    "DIRECTION": "modirect.probing",
    "LETTER": "modirect.probing",
    # io
    "FeatureRef": "modirect.io",
    "load_features": "modirect.io",
    "load_concept_axes": "modirect.io",
    "save_concept_axes": "modirect.io",
    # config
    "Paths": "modirect.config",
    "load_paths": "modirect.config",
    "Direction": "modirect.config",
    "Stage": "modirect.config",
    "canonical_order": "modirect.config",
}

__all__ = ["__version__", *_SUBMODULES, *sorted(_LAZY)]

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs; never runs
    from . import concepts, config, interventions, io, models, probing, tasks
    from .concepts import ConceptAxes, concept_axes_by_layer, extract_concept_vectors
    from .config import Direction, Paths, Stage, canonical_order, load_paths
    from .interventions import (
        CONDITIONS,
        LastTokenIntervention,
        apply_condition,
        last_token_hook,
        project_on_axis,
    )
    from .io import FeatureRef, load_concept_axes, load_features, save_concept_axes
    from .probing import (
        DIRECTION,
        LETTER,
        ProbeConfig,
        ProbeResult,
        ProbeTarget,
        get_preset,
        train_linear_probe,
    )


def __getattr__(name: str) -> Any:
    """Resolve subpackages and re-exported names on first access (PEP 562).

    Each resolved object is cached into `globals()`, so this runs once per name —
    `__getattr__` is only consulted on a lookup miss.
    """
    import importlib

    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _LAZY:
        obj = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the lazily-bound names to `dir()` and tab-completion."""
    return sorted(__all__)
