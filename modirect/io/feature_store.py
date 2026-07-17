"""Reading the cached feature arrays off disk.

`modirect.config.stages` owns the LAYOUT (which directory, which filename); this module
owns the LOADING (open it, attach its labels/meta, check it is not truncated). Paths are
delegated to `config.stages.feature_file` / `labels_file` / `meta_file` rather than rebuilt
here, so the filename convention has exactly one definition.

    {feature_root}/{model}/{stage}/{task}/features.npy            unlayered stages
    {feature_root}/{model}/{stage}/{task}/features_layer_{L}.npy  layered stages
    {feature_root}/{model}/{stage}/{task}/labels.npy              int64 (N,)  — every stage
    {feature_root}/{model}/{stage}/{task}/meta.npy                dict in a .npy
    {feature_root}/{model}/{stage}/{task}/qids.npy                <str> (N,)  — NOT every stage

SIDECAR CORRECTION — qids.npy is not universal
----------------------------------------------
`config/stages.py:35` lists `qids.npy` alongside `labels.npy` for every stage. Verified
against `FeatureWriter.finalize` (`extract_vision_features.py:234-263`), that is not what
the writer does:

    vision_token      labels.npy  qids.npy  meta.npy     :236-239
    vision_encoder    labels.npy            meta.npy     :245-247
    after_projector   labels.npy            meta.npy     :253-255
    after_gate        labels.npy            meta.npy     :261-263

Only `vision_token` gets qids on the vision side (`:237` is the sole `qids.npy` write in
that function); `answer_token` gets its own from a different writer
(`extract_answer_features.py:442`). `labels.npy` genuinely is written into all four, so
config's claim holds for labels and over-reaches for qids.

This matters because the letter probe joins on qid
(`modirect.probing.targets.join_letter_labels`): a letter probe at `after_projector` is
impossible from that directory alone. `qids_for` below borrows the sibling
`vision_token/qids.npy`, which is row-aligned because all four vision stages are filled
from one loop sharing a single write cursor (`FeatureWriter._idx`, advanced once per sample
at `extract_vision_features.py:530`).

STORAGE FORMAT — verified against the extractors, and it is not uniform
----------------------------------------------------------------------
fp16, always. `FeatureWriter._get_mmap:186-188` opens every memmap `dtype=np.float16`, and
the answer path casts at `extract_answer_features.py:242`. Upcast before arithmetic —
`modirect.concepts.axes` accumulates in fp32 for exactly this reason.

2-D on disk, always — but only the vision stages are FLATTENED, and this is the gap
between the stored shape and the shape CLAUDE.md quotes:

    stage             CLAUDE.md "stored shape"    ACTUAL array on disk
    vision_encoder    (N, 8, 1152)                (N, 9216)     = (N, 8*1152)
    after_projector   (N, 8, 3584)                (N, 28672)    = (N, 8*3584)
    vision_token      (N, 8, 3584) per layer      (N, 28672)    per layer
    answer_token      (N, 3584)    per layer      (N, 3584)     — already flat, 1 token

The writer's memmap is 2-D by construction (`shape=(num_samples, feat_dim)`, :187-188) and
every caller flattens first: `layer_stack.reshape(num_layers, -1)` (:501),
`ve_out.reshape(-1)` (:511), `pooled.reshape(-1)` (:519). So the (N, 8, D) in the table is
the LOGICAL shape and `np.load` hands you (N, 8*D). `FeatureRef.unflatten()` recovers it —
and refuses when `--pool_spatial` was off, because then the width is T*P*D and the reshape
would silently fold patches into the frame axis.

Loading is `mmap_mode="r"` by default: a per-layer answer array is ~43 MB, but a sweep is
28 layers x 4 tasks x 3 models, and the legacy `np.load(...).astype(np.float32)`
(`letter_vs_direction_probing.py:100`) is what made those sweeps memory-bound. Memmaps are
read-only; copy before mutating.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config.stages import (
    Stage,
    feature_file,
    labels_file,
    meta_file,
    stage_dir,
)

__all__ = [
    "FeatureRef",
    "load_features",
    "load_labels",
    "load_meta",
    "qids_for",
    "available_layers",
]


def _resolve_root(feature_root: str | os.PathLike | None) -> Path:
    """Default `feature_root` to `Paths.feature_root` (the 8-way cache, Sections A-G).

    Resolved lazily, per call, rather than captured in a module-level constant: `load_paths`
    reads `configs/paths.yaml` and the environment, and binding it at import time would
    freeze whatever the environment happened to be when the module was first touched.

    Letter probing needs `Paths.feature_root_4way` instead — the letter label space only
    exists in the 4-way re-extraction — so pass it explicitly. Nothing here can detect the
    mistake: an 8-way cache probed for letters simply joins against A..H and reports a
    different chance level.
    """
    if feature_root is not None:
        return Path(feature_root)
    from ..config.paths import load_paths

    return load_paths().feature_root


@dataclass(frozen=True)
class FeatureRef:
    """A located feature array plus the metadata needed to interpret it.

    Attributes:
        path: the `.npy` actually loaded.
        features: (N, F) array, fp16 unless `dtype` was passed. May be a read-only memmap.
        labels: (N,) int64 DIRECTION codes in Ordering 1 — decode with
            `modirect.config.directions.to_str`, never by zipping against a hand-written
            name list. These are direction labels even at `answer_token`, despite being
            derived from the MCQ answer letter (see `modirect.probing.targets`).
        meta: the `meta.npy` dict.
        stage: the stage this came from.
        task: full lmms-eval task name.
        model: model name.
        layer: layer index, or None for unlayered stages.
    """

    path: Path
    features: np.ndarray
    labels: np.ndarray
    meta: Mapping[str, Any]
    stage: Stage
    task: str
    model: str
    layer: int | None = None

    @property
    def n_samples(self) -> int:
        """Row count N."""
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        """Stored width F — T*D for the vision stages, D for answer_token."""
        return int(self.features.shape[1])

    @property
    def num_frames(self) -> int:
        """Temporal length T, from `meta["num_frames"]` (`extract_vision_features.py:552`).

        Returns 1 for `answer_token`, which has no T axis and no such key.
        """
        if self.stage is Stage.ANSWER_TOKEN:
            return 1
        return int(self.meta.get("num_frames", 1))

    @property
    def is_flattened(self) -> bool:
        """True when the stored width is T*D rather than D."""
        return self.stage is not Stage.ANSWER_TOKEN and self.num_frames > 1

    def unflatten(self) -> np.ndarray:
        """Recover the logical (N, T, D) view of a flattened vision stage.

        The writer flattened a (T, D) tensor with `reshape(-1)`
        (`extract_vision_features.py:501/511/519`), so the inverse is exactly
        `reshape(N, T, -1)`.

        Returns:
            (N, T, D). For `answer_token`, (N, 1, D).

        Raises:
            ValueError: if extracted without `--pool_spatial` (width is T*P*D and the
                reshape would interleave patches into the frame axis), or if F is not
                divisible by T.
        """
        if not self.is_flattened:
            return self.features.reshape(self.n_samples, 1, -1)
        if not bool(self.meta.get("pool_spatial", False)):
            raise ValueError(
                f"{self.path} was extracted without --pool_spatial "
                f"(tokens_per_frame_post={self.meta.get('tokens_per_frame_post')}); its "
                "width is T*P*D and cannot be reshaped to (N, T, D) unambiguously"
            )
        t = self.num_frames
        if self.feature_dim % t:
            raise ValueError(
                f"feature_dim {self.feature_dim} is not divisible by num_frames {t}"
            )
        return self.features.reshape(self.n_samples, t, self.feature_dim // t)

    def validate(self) -> "FeatureRef":
        """Check rows against labels and `meta["num_samples"]`; return self.

        Worth calling once per sweep: `FeatureWriter.finalize:212-230` TRUNCATES the mmap
        when samples were skipped, so an array left by an interrupted run can outlive its
        labels and would then be probed against a target it is misaligned with.

        Raises:
            ValueError: on any row-count disagreement.
        """
        if len(self.labels) != self.n_samples:
            raise ValueError(
                f"{self.path}: {self.n_samples} rows but {len(self.labels)} labels"
            )
        declared = self.meta.get("num_samples")
        if declared is not None and int(declared) != self.n_samples:
            raise ValueError(
                f"{self.path}: {self.n_samples} rows but meta says {declared} — likely a "
                "truncated array from an interrupted extraction"
            )
        return self


def load_meta(
    model: str,
    stage: Stage | str,
    task: str,
    *,
    feature_root: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Load `meta.npy` — a dict pickled inside a `.npy`, not JSON.

    Needs `allow_pickle=True` and `.item()`; omitting `.item()` yields an unindexable 0-d
    object array rather than an error, which is why every legacy call site spells it out
    (`linear_probe.py:132`).

    Returns:
        Vision keys (`extract_vision_features.py:545-558`): num_classes, label_list,
        model_name, task, num_samples, num_frames, tokens_per_frame_post/pre, hidden_dim,
        vision_hidden_dim, pool_spatial (+ num_layers for vision_token). Answer keys
        (`extract_answer_features.py:451-459`): num_layers, num_samples, num_classes,
        label_list, model_name, task, hidden_dim, token_type="answer".

    Raises:
        FileNotFoundError: naming the path.
    """
    path = meta_file(_resolve_root(feature_root), model, stage, task)
    if not path.exists():
        raise FileNotFoundError(f"no meta.npy at {path}")
    return np.load(path, allow_pickle=True).item()


def load_labels(
    model: str,
    stage: Stage | str,
    task: str,
    *,
    feature_root: str | os.PathLike | None = None,
) -> np.ndarray:
    """Load `labels.npy` — (N,) int64 direction codes in Ordering 1.

    Present in every stage directory (`extract_vision_features.py:236-261` writes the same
    array four times), so any stage is a valid source.
    """
    path = labels_file(_resolve_root(feature_root), model, stage, task)
    if not path.exists():
        raise FileNotFoundError(f"no labels.npy at {path}")
    return np.load(path)


def qids_for(
    model: str,
    stage: Stage | str,
    task: str,
    *,
    feature_root: str | os.PathLike | None = None,
) -> np.ndarray:
    """Load the qids row-aligned with `stage`, borrowing from vision_token when absent.

    See the module docstring: only `vision_token` and `answer_token` carry `qids.npy`. For
    the unlayered vision stages this falls back to `vision_token/qids.npy`, which is
    row-aligned by construction (one shared write cursor, `extract_vision_features.py:530`).

    Returns:
        (N,) array of qid strings, formatted `f"{sample_id}_{direction}"`.

    Raises:
        FileNotFoundError: if neither the stage's own qids nor the fallback exists —
            meaning no letter probe is possible for that extraction.
    """
    root = _resolve_root(feature_root)
    own = stage_dir(root, model, stage, task) / "qids.npy"
    if own.exists():
        return np.load(own, allow_pickle=True)
    fallback = stage_dir(root, model, Stage.VISION_TOKEN, task) / "qids.npy"
    if fallback.exists():
        return np.load(fallback, allow_pickle=True)
    raise FileNotFoundError(
        f"no qids.npy at {own} and no vision_token fallback at {fallback}; "
        f"stage {stage} cannot be joined to letter labels"
    )


def available_layers(
    model: str,
    stage: Stage | str,
    task: str,
    *,
    feature_root: str | os.PathLike | None = None,
) -> list[int]:
    """List the layer indices actually present on disk, ascending.

    Globs rather than trusting `meta["num_layers"]`: an interrupted extraction leaves meta
    claiming 29 layers while only some files exist.

    Returns:
        Sorted layer indices; empty for an unlayered stage or a missing directory.
    """
    directory = stage_dir(_resolve_root(feature_root), model, stage, task)
    if not directory.is_dir():
        return []
    layers = []
    for p in directory.glob("features_layer_*.npy"):
        try:
            layers.append(int(p.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):  # not one of ours
            continue
    return sorted(layers)


def load_features(
    model: str,
    stage: Stage | str,
    task: str,
    layer: int | None = None,
    *,
    feature_root: str | os.PathLike | None = None,
    mmap: bool = True,
    dtype: np.dtype | None = None,
    validate: bool = True,
) -> FeatureRef:
    """Load one cached feature array with its labels and meta.

    Args:
        model: model name, e.g. `llava-video-7b_lora_4combo_v2_delta`.
        stage: a `Stage` or its string value.
        task: full lmms-eval task name, e.g.
            `vlm_direction_testbed_R2R_4way_1500_obj_place`.
        layer: required for layered stages, rejected for unlayered ones — `config.stages`
            enforces this, so `layer=21` on `vision_encoder` raises rather than silently
            returning the wrong stage's tensor.
        feature_root: defaults to `Paths.feature_root` (8-way, Sections A-G). Pass
            `load_paths().feature_root_4way` for anything involving letters.
        mmap: load with `mmap_mode="r"` (default). Result is READ-ONLY; pass `mmap=False`
            or a `dtype` for a writable copy.
        dtype: upcast on load, e.g. `np.float32`. Forces a full read into RAM. Prefer
            leaving it None and letting `modirect.concepts.axes` upcast per batch.
        validate: run `FeatureRef.validate()`.

    Returns:
        FeatureRef.

    Raises:
        ValueError: on a layer/stage mismatch, or a layer outside L0..L27.
        FileNotFoundError: naming the exact missing path.

    Example:
        >>> ref = load_features("baseline", "answer_token", "task", layer=21)  # doctest: +SKIP
        >>> ref.features.shape                                                 # doctest: +SKIP
        (6000, 3584)
    """
    stage = Stage(stage)
    root = _resolve_root(feature_root)
    path = feature_file(root, model, stage, task, layer)

    if not path.exists():
        directory = path.parent
        extra = "" if directory.is_dir() else f" (stage directory {directory} does not exist)"
        if directory.is_dir() and stage.is_layered:
            present = available_layers(model, stage, task, feature_root=root)
            extra = f" (layers present: {present or 'none'})"
        raise FileNotFoundError(f"no features at {path}{extra}")

    features = np.load(path, mmap_mode="r" if mmap and dtype is None else None)
    if dtype is not None:
        features = np.asarray(features, dtype=dtype)

    ref = FeatureRef(
        path=path,
        features=features,
        labels=load_labels(model, stage, task, feature_root=root),
        meta=load_meta(model, stage, task, feature_root=root),
        stage=stage,
        task=task,
        model=model,
        layer=layer,
    )
    return ref.validate() if validate else ref
