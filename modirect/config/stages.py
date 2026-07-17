"""The four pipeline stages features are cached at, their shapes, and their disk layout.

Vision encoder -> projector -> LLM is the axis the whole project measures along, and the
cache mirrors it. Shapes are quoted from `CLAUDE.md` "Feature Shapes & Pooling":

    Stage             Stored shape            Meaning
    vision_encoder    (N, 8, 1152)            SigLIP output, spatial mean-pooled
    after_projector   (N, 8, 3584)            mm_projector + bilinear pool, then spatial mean
    vision_token      (N, 8, 3584) per layer  LLM decoder layer l, vision positions, spatial mean
    answer_token      (N, 3584)    per layer  LLM decoder layer l, last token

The constants behind those numbers are not free parameters:

    T = 8     `max_frames_num=8` in `modirect.config.registry.VANILLA_ARGS`
    1152      SigLIP hidden size
    3584      Qwen2-7B hidden size (`D_llm`)
    L = 28    decoder layers, L0..L27 (`mechanism_diagnosis.py:43  ALL_LAYERS = range(28)`)

THE POOLING RULE (CLAUDE.md, "규칙"): for vision-side probing the N(spatial) axis is
dropped and the T(temporal) axis is kept. This is applied at *extraction* time via
`--pool_spatial` (`linear_probing/extract_vision_features.py`), not at analysis time — so
a cache written without that flag has a different, incompatible shape, and its
`meta.npy` records `pool_spatial` to disambiguate. `answer_token` has no T axis at all:
it is a single last-token position, which is why it alone is (N, D).

Only `vision_token` and `answer_token` are layered; the two upstream stages are single
tensors. That asymmetry is the reason for the two filename forms below.

ON-DISK LAYOUT (`CLAUDE.md` "디렉토리 구조"; written by
`extract_vision_features.py:225-261 FeatureWriter.finalize`):

    {feature_root}/{model}/{stage}/{task}/features.npy            # unlayered stages
    {feature_root}/{model}/{stage}/{task}/features_layer_{l}.npy  # layered stages
    {feature_root}/{model}/{stage}/{task}/labels.npy              # int64, ALWAYS present
    {feature_root}/{model}/{stage}/{task}/qids.npy
    {feature_root}/{model}/{stage}/{task}/meta.npy                # dict, np.save'd

`labels.npy` is written identically into every stage directory
(`extract_vision_features.py:236-261` saves the same `labels_array` four times), so any
stage can be read standalone. Those integers follow Ordering 1 — see
`modirect.config.directions`, which documents why that matters.

`meta.npy` is a **`.npy` holding a dict**, not JSON; load it with
`np.load(..., allow_pickle=True).item()`. It carries `num_classes`, `label_list`,
`num_samples`, `hidden_dim`, `pool_spatial` (`extract_vision_features.py:544-558`).

A note on `{task}`: directory names are the full lmms-eval task name, not the short one
used in CLAUDE.md's tables — e.g. `vlm_direction_testbed_R2R_4way_1500_obj_place`, per
`analysis/task_invariance/axis_layer_cos.py:21`. `stage_dir()` takes whatever string you
give it; `modirect.config` does not own the task-name registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
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

#: Qwen2-7B decoder layers, L0..L27. `mechanism_diagnosis.py:43`.
N_LLM_LAYERS: int = 28
LLM_LAYERS: tuple[int, ...] = tuple(range(N_LLM_LAYERS))

#: T — frames per sample. Pinned by `max_frames_num=8` in `VANILLA_ARGS`.
N_FRAMES: int = 8
#: SigLIP hidden size.
D_VISION: int = 1152
#: Qwen2-7B hidden size.
D_LLM: int = 3584


class Stage(str, Enum):
    """A point in the vision -> projector -> LLM pipeline where features are cached.

    Values are the on-disk directory names, so `f"{Stage.ANSWER_TOKEN}"`-style path
    building and `Stage("answer_token")` round-tripping both work.
    """

    VISION_ENCODER = "vision_encoder"
    AFTER_PROJECTOR = "after_projector"
    VISION_TOKEN = "vision_token"
    ANSWER_TOKEN = "answer_token"

    def __str__(self) -> str:  # py3.10 `str, Enum` would render "Stage.VISION_ENCODER"
        return self.value

    @property
    def spec(self) -> "StageSpec":
        """The `StageSpec` describing this stage's shape and layering."""
        return STAGE_SPECS[self]

    @property
    def is_layered(self) -> bool:
        """True if this stage stores one file per LLM layer."""
        return self.spec.layered


@dataclass(frozen=True)
class StageSpec:
    """Static description of one cache stage.

    Attributes:
        stage: The stage this describes.
        shape: Stored shape as documented in CLAUDE.md, per file. For layered stages
            this is the shape of a single `features_layer_{l}.npy`.
        layered: True if there is one file per LLM layer, False for a single
            `features.npy`.
        has_temporal: True if the stored array keeps the T=8 frame axis. False only for
            `answer_token`, which is a single token position.
        dim: The trailing feature dimension.
        description: What the tensor is, and where it sits in the pipeline.
    """

    stage: Stage
    shape: tuple[int | str, ...]
    layered: bool
    has_temporal: bool
    dim: int
    description: str


STAGE_SPECS: dict[Stage, StageSpec] = {
    Stage.VISION_ENCODER: StageSpec(
        stage=Stage.VISION_ENCODER,
        shape=("N", N_FRAMES, D_VISION),
        layered=False,
        has_temporal=True,
        dim=D_VISION,
        description=(
            "SigLIP output, spatial mean-pooled. Frozen — identical across all three "
            "models, which is why the Vision Encoder row of every CLAUDE.md table shows "
            "one number for Vanilla/Baseline/Delta."
        ),
    ),
    Stage.AFTER_PROJECTOR: StageSpec(
        stage=Stage.AFTER_PROJECTOR,
        shape=("N", N_FRAMES, D_LLM),
        layered=False,
        has_temporal=True,
        dim=D_LLM,
        description=(
            "mm_projector output after bilinear pooling, spatial mean-pooled. The stage "
            "the delta_direct auxiliary loss acts on, and the only stage where Delta "
            "diverges from Baseline before the LLM runs."
        ),
    ),
    Stage.VISION_TOKEN: StageSpec(
        stage=Stage.VISION_TOKEN,
        shape=("N", N_FRAMES, D_LLM),
        layered=True,
        has_temporal=True,
        dim=D_LLM,
        description=(
            "LLM decoder layer l, hidden states at the vision positions, spatial "
            "mean-pooled. One file per layer, L0..L27."
        ),
    ),
    Stage.ANSWER_TOKEN: StageSpec(
        stage=Stage.ANSWER_TOKEN,
        shape=("N", D_LLM),
        layered=True,
        has_temporal=False,
        dim=D_LLM,
        description=(
            "LLM decoder layer l, last-token hidden state — the readout position. No T "
            "axis. This is where the L21 canonical direction axis and the L16-L17 letter "
            "binding are measured."
        ),
    ),
}


def stage_dir(feature_root: Path | str, model: str, stage: Stage | str, task: str) -> Path:
    """Build `{feature_root}/{model}/{stage}/{task}`.

    Args:
        feature_root: `Paths.feature_root` (8-way, Sections A-G) or
            `Paths.feature_root_4way` (Section H — the only cache with letter labels).
        model: A short name from `modirect.config.registry.MODEL_NAMES`.
        stage: A `Stage` or its string value.
        task: Full lmms-eval task name, e.g.
            `vlm_direction_testbed_R2R_4way_1500_obj_place`.

    Returns:
        The directory path. Not checked for existence — the feature caches are absent on
        this host (see `modirect.config.paths`).
    """
    return Path(feature_root) / model / str(Stage(stage)) / task


def feature_file(
    feature_root: Path | str,
    model: str,
    stage: Stage | str,
    task: str,
    layer: int | None = None,
) -> Path:
    """Resolve the `.npy` holding one stage's features.

    Picks `features.npy` vs `features_layer_{l}.npy` from the stage's `layered` flag, so
    callers do not re-derive the filename convention (it is currently written out by
    hand at each read site, e.g. `axis_layer_cos.py:22`).

    Args:
        feature_root: See `stage_dir`.
        model: See `stage_dir`.
        stage: See `stage_dir`.
        task: See `stage_dir`.
        layer: Required for layered stages (`vision_token`, `answer_token`); must be
            omitted for the unlayered ones.

    Returns:
        Path to the feature array.

    Raises:
        ValueError: if `layer` is missing for a layered stage, supplied for an unlayered
            one, or outside L0..L27. Passing a layer to `vision_encoder` is a category
            error — there is only one tensor — and silently ignoring it would return the
            wrong stage's data without complaint.

    Example:
        >>> feature_file("/f", "baseline", Stage.ANSWER_TOKEN, "t", 21).name
        'features_layer_21.npy'
    """
    stage = Stage(stage)
    directory = stage_dir(feature_root, model, stage, task)
    if not stage.is_layered:
        if layer is not None:
            raise ValueError(
                f"stage {stage} is not layered; it stores a single features.npy, "
                f"but layer={layer} was given")
        return directory / "features.npy"
    if layer is None:
        raise ValueError(f"stage {stage} is layered; a layer in 0..{N_LLM_LAYERS - 1} is required")
    if not 0 <= layer < N_LLM_LAYERS:
        raise ValueError(f"layer {layer} out of range 0..{N_LLM_LAYERS - 1}")
    return directory / f"features_layer_{layer}.npy"


def labels_file(feature_root: Path | str, model: str, stage: Stage | str, task: str) -> Path:
    """Resolve `labels.npy` for a stage.

    The same int64 array is written into every stage directory
    (`extract_vision_features.py:236-261`), so any stage is a valid source. Values follow
    Ordering 1 — decode them with `modirect.config.directions.to_str`, never by zipping
    against a hand-written name list.
    """
    return stage_dir(feature_root, model, stage, task) / "labels.npy"


def meta_file(feature_root: Path | str, model: str, stage: Stage | str, task: str) -> Path:
    """Resolve `meta.npy` for a stage.

    It is a pickled dict inside a `.npy`, not JSON:
    `np.load(meta_file(...), allow_pickle=True).item()` yields `num_classes`,
    `label_list`, `num_samples`, `hidden_dim`, `pool_spatial`.
    """
    return stage_dir(feature_root, model, stage, task) / "meta.npy"
