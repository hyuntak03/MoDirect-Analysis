"""The three model short names, and how to turn one into an lmms-eval args string.

`vanilla` / `baseline` / `delta` are the axis of nearly every table in CLAUDE.md, and
every runtime script re-declares them. The declarations are byte-identical copies:

    analysis/task_invariance/mechanism_diagnosis.py:36-37     VANILLA_ARGS, BASELINE_LORA
    analysis/task_invariance/extract_attn_mlp_contrib.py:32-34  VANILLA_ARGS, BASELINE_LORA, DELTA_LORA
    analysis/task_invariance/vision_intervention_v2.py:37-38    BASELINE_LORA, DELTA_LORA

and the composition rule is copied too — `extract_attn_mlp_contrib.py:74-78`:

    if vanilla:   model_args_str = VANILLA_ARGS
    elif baseline: model_args_str = f"lora_pretrained={BASELINE_LORA},{VANILLA_ARGS}"
    elif delta:    model_args_str = f"lora_pretrained={DELTA_LORA},{VANILLA_ARGS}"

`resolve_model_args` is that rule, once. The prefix ordering is preserved exactly:
`lora_pretrained` FIRST, then the vanilla args. `core/model_loader.parse_model_args`
splits on commas, so a reordering would be harmless there — but the string is also what
gets logged and used to name result directories, and existing results on disk were
written under the current ordering.

WHAT IS AND IS NOT HARDCODED HERE
    The LoRA **directory basenames** are verbatim, because they are the identity of the
    trained artefact — they encode the recipe (`r64_f8_ep1_lr1e-5`) and changing one
    means a different model, not a different host.
    The LoRA **root** is not: it comes from `Paths.checkpoint_root`. The source repo
    hardcoded an absolute `/data/takhyun03/.../LLaVA-NeXT/work_dirs/...`
    (`mechanism_diagnosis.py:37`) which does not exist on this host; the migrated
    scripts already moved to `{_VLM_ROOT}/LLaVA-NeXT/work_dirs/<basename>`
    (`pipeline/05_intervene/delta_effect_analysis.py:42`), and this module matches them.

`vanilla` has no LoRA — it is the stock `lmms-lab/LLaVA-Video-7B-Qwen2` checkpoint, which
is why `lora_dirname is None` is the discriminant rather than a sentinel path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .paths import Paths, load_paths

__all__ = [
    "ModelSpec",
    "MODEL_REGISTRY",
    "MODEL_NAMES",
    "VANILLA_ARGS",
    "resolve_model_args",
    "resolve_lora_path",
    "get_model_spec",
]

#: The stock-model args string, verbatim from `mechanism_diagnosis.py:36` and
#: `extract_attn_mlp_contrib.py:32`. Every model — LoRA or not — is loaded with these;
#: a LoRA run just prepends `lora_pretrained=`.
#:
#: `max_frames_num=8` is why every cached feature has T=8 (CLAUDE.md "Feature Shapes"),
#: and `mm_spatial_pool_mode=bilinear` is the pooling that `after_projector` features are
#: taken downstream of. Neither is a free parameter: changing them invalidates the
#: caches under `Paths.feature_root`.
VANILLA_ARGS = (
    "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,"
    "video_decode_backend=decord,"
    "conv_template=qwen_1_5,"
    "mm_spatial_pool_mode=bilinear,"
    "max_frames_num=8,"
    "device_map=auto,"
    "force_sample=True"
)


@dataclass(frozen=True)
class ModelSpec:
    """One row of the registry.

    Attributes:
        name: Short name — the key used in result paths and CLAUDE.md tables.
        lora_dirname: Basename under `Paths.checkpoint_root`, or None for `vanilla`.
            Verbatim from the source scripts; see the module docstring.
        description: What the model is, in the project's own terms.
    """

    name: str
    lora_dirname: str | None
    description: str


#: The three models. Order is the canonical presentation order used by every table in
#: CLAUDE.md (Vanilla -> Baseline -> Delta = increasing supervision).
MODEL_REGISTRY: Mapping[str, ModelSpec] = {
    "vanilla": ModelSpec(
        name="vanilla",
        lora_dirname=None,
        description=(
            "Stock LLaVA-Video-7B-Qwen2, no fine-tuning. The control: it has no L19 "
            "direction amplifier at all (push ~0 vs Baseline +37.8), and never encodes "
            "letter (probe stays at chance)."
        ),
    ),
    "baseline": ModelSpec(
        name="baseline",
        # extract_attn_mlp_contrib.py:33
        lora_dirname="llava-video-7b-qwen2_baseline_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
        description=(
            "4combo_v2 LoRA trained with the MCQ loss only. Learns the L19 amplifier "
            "and the task-invariant L21 axis (cross-task cos 0.94), but leaves the "
            "projector's direction axis task-specific."
        ),
    ),
    "delta": ModelSpec(
        name="delta",
        # extract_attn_mlp_contrib.py:34
        lora_dirname="llava-video-7b-qwen2_delta_direct_shape_simple_new_lora-r64_f8_ep1_lr1e-5",
        description=(
            "4combo_v2 + delta_direct auxiliary loss on the projector's temporal delta. "
            "Same mechanism as Baseline, fired harder: binding starts one layer earlier "
            "(L16 vs L17) and the projector axis aligns (cos 0.27 -> 0.70). The auxiliary "
            "head is removed at inference, so the load args are identical in shape to "
            "Baseline's — only the weights differ."
        ),
    ),
}

#: Canonical iteration order; `list(MODEL_REGISTRY)` with the intent made explicit.
MODEL_NAMES: tuple[str, ...] = tuple(MODEL_REGISTRY)


def get_model_spec(name: str) -> ModelSpec:
    """Look up a `ModelSpec` by short name.

    Args:
        name: One of `MODEL_NAMES`.

    Returns:
        The registered spec.

    Raises:
        KeyError: with the valid names listed. Scripts take this name from argparse, so
            a typo must not fall through to a default model.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; valid: {list(MODEL_NAMES)}") from None


def resolve_lora_path(name: str, paths: Paths | None = None) -> Path | None:
    """Resolve a model's LoRA directory against `Paths.checkpoint_root`.

    Args:
        name: One of `MODEL_NAMES`.
        paths: Resolved paths. Defaults to `load_paths()`.

    Returns:
        Absolute path to the LoRA directory, or None for `vanilla`, which has none.
        Existence is not checked — the LLaVA-NeXT tree is absent on this host, and this
        function must stay usable for building/inspecting args strings offline.
    """
    spec = get_model_spec(name)
    if spec.lora_dirname is None:
        return None
    paths = paths or load_paths()
    return paths.checkpoint_root / spec.lora_dirname


def resolve_model_args(name: str, paths: Paths | None = None) -> str:
    """Compose the lmms-eval args string for a model, as the scripts do.

    Reproduces `extract_attn_mlp_contrib.py:74-78` exactly: `vanilla` is bare
    `VANILLA_ARGS`; the others are `f"lora_pretrained={lora},{VANILLA_ARGS}"`.

    Args:
        name: One of `MODEL_NAMES`.
        paths: Resolved paths, for the LoRA root. Defaults to `load_paths()`.

    Returns:
        A comma-separated args string for `core.model_loader.parse_model_args`.

    Example:
        >>> resolve_model_args("vanilla") == VANILLA_ARGS
        True
    """
    lora = resolve_lora_path(name, paths)
    if lora is None:
        return VANILLA_ARGS
    return f"lora_pretrained={lora},{VANILLA_ARGS}"
