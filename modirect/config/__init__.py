"""Configuration: paths, models, direction labels, and cache stages.

This package is the answer to "where is that path / what is label 2 / which LoRA is
`delta`" — questions currently answered by 51 hardcoded literals scattered across the
analysis scripts.

IMPORT CONTRACT — this package must stay importable on a host with no llava, no torch,
and no datasets. `core/model_loader.py` imports llava at module scope, and llava is not
installed here, so nothing under `modirect/` may import `core` at module import time;
runtime wrappers do it lazily inside functions. Nothing here touches the filesystem on
import either: `load_paths()` is a call, not a module-level constant, and no path is
validated on construction (most are genuinely absent — see `paths` for the verified
inventory). The only third-party imports are numpy and, lazily, pyyaml.

Start here:
    `load_paths()`            resolve every root (yaml > env > documented defaults)
    `resolve_model_args()`    short name -> lmms-eval args string
    `canonical_order()`       the class order matching `labels.npy`
    `feature_file()`          -> {feature_root}/{model}/{stage}/{task}/features*.npy

If you read one docstring, make it `modirect.config.directions`: the repo stores the
four directions under three mutually incompatible orderings, and mixing them up
mislabels every class without raising.
"""

from __future__ import annotations

from .directions import (
    CHANCE_ACCURACY,
    DIRECTION_CLASSES_4WAY,
    DIRECTION_LABELS_4WAY,
    DIRECTION_NAMES,
    DIRS_FACTORIAL_CLOCKWISE,
    Direction,
    N_DIRECTIONS,
    as_ints,
    as_strs,
    canonical_order,
    reorder_to,
    to_int,
    to_label,
    to_str,
)
from .paths import (
    DEFAULT_CONFIG_RELPATH,
    DEFAULT_VLM_DIRECTION_ROOT,
    Paths,
    load_paths,
    repo_root,
)
from .registry import (
    MODEL_NAMES,
    MODEL_REGISTRY,
    VANILLA_ARGS,
    ModelSpec,
    get_model_spec,
    resolve_lora_path,
    resolve_model_args,
)
from .stages import (
    D_LLM,
    D_VISION,
    LLM_LAYERS,
    N_FRAMES,
    N_LLM_LAYERS,
    STAGE_SPECS,
    Stage,
    StageSpec,
    feature_file,
    labels_file,
    meta_file,
    stage_dir,
)

__all__ = [
    # paths
    "Paths",
    "load_paths",
    "repo_root",
    "DEFAULT_VLM_DIRECTION_ROOT",
    "DEFAULT_CONFIG_RELPATH",
    # registry
    "ModelSpec",
    "MODEL_REGISTRY",
    "MODEL_NAMES",
    "VANILLA_ARGS",
    "get_model_spec",
    "resolve_lora_path",
    "resolve_model_args",
    # directions
    "Direction",
    "DIRECTION_NAMES",
    "N_DIRECTIONS",
    "CHANCE_ACCURACY",
    "canonical_order",
    "to_str",
    "to_int",
    "to_label",
    "as_ints",
    "as_strs",
    "reorder_to",
    "DIRS_FACTORIAL_CLOCKWISE",
    "DIRECTION_CLASSES_4WAY",
    "DIRECTION_LABELS_4WAY",
    # stages
    "Stage",
    "StageSpec",
    "STAGE_SPECS",
    "N_LLM_LAYERS",
    "LLM_LAYERS",
    "N_FRAMES",
    "D_VISION",
    "D_LLM",
    "stage_dir",
    "feature_file",
    "labels_file",
    "meta_file",
]
