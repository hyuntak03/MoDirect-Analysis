"""Direction concept vectors:  Δ_d = mean(h | direction = d) − mean(h | all).

This is the single definition of the project's central quantity. It was previously
re-derived independently in at least ten places, each with slightly different naming
and normalisation:

    analysis/task_invariance/mechanism_diagnosis.py:63   Delta = h_avg_d[d] - g
    analysis/task_invariance/axis_layer_cos.py:30        h[y == d].mean(0) - g
    analysis/task_invariance/analyze_3models.py:30       Delta = proto - g[None, :]
    analysis/task_invariance/magnitude_cascade.py:42     norm(h[y == d].mean(0) - g)
    analysis/task_invariance/measure_subspace_offaxis.py:46
    analysis/task_invariance/stage_trajectory.py:44
    analysis/task_invariance/lm_head_align.py:57
    analysis/task_invariance/vision_token_axis_per_layer.py:51
    analysis/task_invariance/analyze_attn_mlp_contrib.py:38
    analysis/direction_steering.py:83                    class_mean - global_mean

`mechanism_diagnosis.py:55-67` is the reference implementation: it is the only one that
computes every layer at once and returns g / Δ_d / Δ̂_d / ‖Δ_d‖ together. This module
generalises it to any label set and any stage.

Terminology (fixed here, and used across `modirect`):

    g          global mean, mean(h | all)             "the origin"
    delta      Δ_d  = mean(h | d) − g                 the concept vector (RAW, unnormalised)
    delta_hat  Δ̂_d = Δ_d / ‖Δ_d‖                      the direction axis (UNIT)
    mag        ‖Δ_d‖                                  the magnitude carried along that axis

The magnitude is the subject of the project's central finding, so `delta` is deliberately
kept raw: `delta_hat` and `mag` are derived views, never the stored form.
"""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import numpy as np

from .types import ConceptAxes

__all__ = ["extract_concept_vectors", "concept_axes_by_layer"]

_EPS = 1e-9


def extract_concept_vectors(
    features: np.ndarray,
    labels: Sequence[Hashable],
    *,
    classes: Sequence[Hashable] | None = None,
    dtype: np.dtype = np.float32,
) -> ConceptAxes:
    """Compute Δ_d = mean(h | label == d) − mean(h | all) for every class d.

    Args:
        features: (N, ...) array. The leading axis is samples; every trailing axis is
            preserved untouched, so this works for
              (N, D)      a single stage / layer      -> delta[d] is (D,)
              (N, L, D)   all layers at once          -> delta[d] is (L, D)
              (N, T, D)   temporal, unpooled          -> delta[d] is (T, D)
            The (N, L, D) form reproduces `mechanism_diagnosis.compute_stats` exactly.
        labels: length-N sequence of class labels, aligned with `features`.
        classes: class order to use. Defaults to sorted(unique(labels)). Pass this
            explicitly to pin the ordering — never rely on the incidental order of the
            data, because Δ̂ sign conventions depend on it.
        dtype: accumulation dtype. float32 by default; features are commonly stored
            fp16 on disk, and averaging thousands of fp16 samples in fp16 loses signal.

    Returns:
        ConceptAxes with `g`, `delta`, `delta_hat`, `mag`.

    Raises:
        ValueError: if labels and features disagree in length, or a class is absent.

    Note:
        Δ_d is defined against the GLOBAL mean, not against the other classes. With
        balanced classes sum_d Δ_d == 0, so the Δ vectors are linearly dependent — that
        is expected and is why 4 directions span a 3-dimensional subspace (see
        `modirect.geometry.subspace`).
    """
    features = np.asarray(features)
    if features.ndim < 2:
        raise ValueError(f"features must be at least 2-D (N, ...), got {features.shape}")

    labels_arr = np.asarray(labels)
    if len(labels_arr) != len(features):
        raise ValueError(
            f"labels length {len(labels_arr)} != features length {len(features)}")

    if classes is None:
        classes = sorted(np.unique(labels_arr).tolist())

    work = features.astype(dtype, copy=False)
    g = work.mean(axis=0)

    delta: dict[Hashable, np.ndarray] = {}
    for d in classes:
        mask = labels_arr == d
        n = int(mask.sum())
        if n == 0:
            raise ValueError(
                f"class {d!r} has no samples; pass `classes` to pin the label set, "
                f"or drop the class. Present: {sorted(np.unique(labels_arr).tolist())}")
        delta[d] = work[mask].mean(axis=0) - g

    # ‖·‖ over the feature axis only, so (L, D) -> (L,) and (D,) -> scalar.
    mag = {d: np.linalg.norm(v, axis=-1) for d, v in delta.items()}
    delta_hat = {
        d: v / (np.linalg.norm(v, axis=-1, keepdims=True) + _EPS)
        for d, v in delta.items()
    }

    return ConceptAxes(
        g=g,
        delta=delta,
        delta_hat=delta_hat,
        mag=mag,
        classes=tuple(classes),
        n_samples={d: int((labels_arr == d).sum()) for d in classes},
    )


def concept_axes_by_layer(
    features: np.ndarray,
    labels: Sequence[Hashable],
    *,
    classes: Sequence[Hashable] | None = None,
) -> Mapping[int, ConceptAxes]:
    """Split an (N, L, D) stack into one ConceptAxes per layer.

    `extract_concept_vectors` on an (N, L, D) array already keeps the layer axis, which
    is what you want for vectorised work. Use this instead when you need to hand a
    single layer's axes to something that expects flat (D,) vectors — e.g.
    `modirect.interventions.operators`, which acts on one layer's last-token slice.
    """
    features = np.asarray(features)
    if features.ndim != 3:
        raise ValueError(f"expected (N, L, D), got {features.shape}")
    return {
        layer: extract_concept_vectors(features[:, layer, :], labels, classes=classes)
        for layer in range(features.shape[1])
    }
