"""Tests for `modirect.concepts.axes` — the Δ_d = mean(h|d) − mean(h|all) definition.

These are numeric-recovery tests, not smoke tests: the fixtures are built so that the
correct answer is known in closed form and can be asserted exactly (up to float32
round-off), rather than merely asserting that the code returns *something* of the right
shape.

numpy only — no torch, no llava. See `modirect/__init__.py` for why that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from modirect.concepts import ConceptAxes, concept_axes_by_layer, extract_concept_vectors

# float32 accumulation (axes.py:93 casts to float32 by default) over a few hundred
# samples; 1e-5 is comfortably above the round-off floor and far below any real signal.
TOL = 1e-5

CLASSES = ("Up", "Down", "Left", "Right")


def _known_deltas(dim: int, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Four Δ vectors that sum to exactly zero.

    Sum-zero is what makes the fixture exactly invertible: with balanced classes the
    global mean g equals the base vector, so the recovered Δ_d must equal the injected
    one with no residual to explain away.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(len(CLASSES), dim)).astype(np.float64)
    raw -= raw.mean(axis=0, keepdims=True)  # force sum_d Δ_d == 0
    return {c: raw[i] for i, c in enumerate(CLASSES)}


def _synthetic(dim: int = 16, pairs: int = 6, *, seed: int = 0):
    """Build (features, labels, g0, deltas) with per-class means known exactly.

    Within each class the noise is injected in ± pairs, so the class mean is exactly
    g0 + Δ_d — the sample noise cancels by construction instead of merely averaging
    out. That turns "recovers the delta" into an exact assertion.
    """
    rng = np.random.default_rng(seed + 1)
    deltas = _known_deltas(dim, seed=seed)
    g0 = rng.normal(size=dim).astype(np.float64) * 3.0

    feats, labels = [], []
    for c in CLASSES:
        noise = rng.normal(size=(pairs, dim))
        block = np.concatenate([g0 + deltas[c] + noise, g0 + deltas[c] - noise])
        feats.append(block)
        labels.extend([c] * (2 * pairs))
    return np.concatenate(feats), np.array(labels), g0, deltas


def test_recovers_known_delta_per_class():
    """Δ_d and g come back as injected."""
    features, labels, g0, deltas = _synthetic()
    axes = extract_concept_vectors(features, labels, classes=CLASSES)

    np.testing.assert_allclose(axes.g, g0, atol=TOL)
    for c in CLASSES:
        np.testing.assert_allclose(axes.delta[c], deltas[c], atol=TOL)


def test_deltas_sum_to_zero_for_balanced_classes():
    """sum_d Δ_d == 0 — the Δ vectors are linearly dependent by construction.

    Δ is measured against the GLOBAL mean, so with balanced classes the class means
    average back to g. This is why 4 directions span a 3-dimensional subspace
    (axes.py:75-79) and why any rank-4 assumption downstream is wrong.
    """
    features, labels, _, _ = _synthetic()
    axes = extract_concept_vectors(features, labels, classes=CLASSES)

    total = np.sum([axes.delta[c] for c in CLASSES], axis=0)
    np.testing.assert_allclose(total, np.zeros(axes.dim), atol=TOL)


def test_delta_hat_is_unit_and_factorises_delta():
    """‖Δ̂_d‖ == 1 and Δ_d == ‖Δ_d‖ · Δ̂_d.

    The magnitude/axis split is the project's central decomposition: `delta` stays raw
    because ‖Δ‖ is the finding, while `delta_hat` is the derived unit view
    (axes.py:29-30). This asserts the two views are consistent.
    """
    features, labels, _, _ = _synthetic()
    axes = extract_concept_vectors(features, labels, classes=CLASSES)

    for c in CLASSES:
        np.testing.assert_allclose(np.linalg.norm(axes.delta_hat[c]), 1.0, atol=TOL)
        np.testing.assert_allclose(
            axes.mag[c] * axes.delta_hat[c], axes.delta[c], atol=TOL)


def test_magnitude_matches_norm_of_delta():
    """mag is exactly ‖Δ‖ over the feature axis."""
    features, labels, _, deltas = _synthetic()
    axes = extract_concept_vectors(features, labels, classes=CLASSES)

    for c in CLASSES:
        np.testing.assert_allclose(
            axes.mag[c], np.linalg.norm(deltas[c]), atol=TOL)


def test_layer_stacked_equals_per_layer():
    """(N, L, D) in one shot == per-layer (N, D) computation.

    This is the `mechanism_diagnosis.compute_stats` equivalence (axes.py:61): the
    vectorised all-layers path must not differ from the loop the original scripts ran,
    or every cross-layer number in the paper shifts.
    """
    n_layers, dim = 5, 12
    per_layer = [_synthetic(dim=dim, seed=10 + i) for i in range(n_layers)]
    labels = per_layer[0][1]
    stacked = np.stack([f for f, _, _, _ in per_layer], axis=1)  # (N, L, D)
    assert stacked.shape == (len(labels), n_layers, dim)

    axes = extract_concept_vectors(stacked, labels, classes=CLASSES)
    assert axes.g.shape == (n_layers, dim)

    for layer in range(n_layers):
        flat = extract_concept_vectors(
            per_layer[layer][0], labels, classes=CLASSES)
        np.testing.assert_allclose(axes.g[layer], flat.g, atol=TOL)
        for c in CLASSES:
            np.testing.assert_allclose(axes.delta[c][layer], flat.delta[c], atol=TOL)
            np.testing.assert_allclose(
                axes.delta_hat[c][layer], flat.delta_hat[c], atol=TOL)
            np.testing.assert_allclose(axes.mag[c][layer], flat.mag[c], atol=TOL)


def test_mag_shape_is_per_layer_for_stacked_input():
    """mag is (L,) for layer-stacked input, scalar otherwise (types.py:25)."""
    n_layers, dim = 4, 8
    per_layer = [_synthetic(dim=dim, seed=20 + i) for i in range(n_layers)]
    labels = per_layer[0][1]
    stacked = np.stack([f for f, _, _, _ in per_layer], axis=1)

    axes = extract_concept_vectors(stacked, labels, classes=CLASSES)
    assert axes.mag["Up"].shape == (n_layers,)

    flat = extract_concept_vectors(per_layer[0][0], labels, classes=CLASSES)
    assert np.ndim(flat.mag["Up"]) == 0


def test_at_layer_slices_stacked_axes():
    """at_layer(l) reproduces the standalone single-layer extraction.

    Interventions act on one layer's (D,) vectors, so this slice is the bridge between
    the vectorised form and `modirect.interventions.operators`.
    """
    n_layers, dim = 3, 10
    per_layer = [_synthetic(dim=dim, seed=30 + i) for i in range(n_layers)]
    labels = per_layer[0][1]
    stacked = np.stack([f for f, _, _, _ in per_layer], axis=1)

    axes = extract_concept_vectors(stacked, labels, classes=CLASSES)
    sliced = axes.at_layer(1)
    flat = extract_concept_vectors(per_layer[1][0], labels, classes=CLASSES)

    assert sliced.dim == dim
    np.testing.assert_allclose(sliced.g, flat.g, atol=TOL)
    for c in CLASSES:
        np.testing.assert_allclose(sliced.delta[c], flat.delta[c], atol=TOL)


def test_concept_axes_by_layer_matches_stacked():
    """The dict-of-layers helper agrees with the stacked computation."""
    n_layers, dim = 3, 9
    per_layer = [_synthetic(dim=dim, seed=40 + i) for i in range(n_layers)]
    labels = per_layer[0][1]
    stacked = np.stack([f for f, _, _, _ in per_layer], axis=1)

    by_layer = concept_axes_by_layer(stacked, labels, classes=CLASSES)
    axes = extract_concept_vectors(stacked, labels, classes=CLASSES)

    assert sorted(by_layer) == list(range(n_layers))
    for layer, one in by_layer.items():
        np.testing.assert_allclose(one.g, axes.g[layer], atol=TOL)
        for c in CLASSES:
            np.testing.assert_allclose(one.delta[c], axes.delta[c][layer], atol=TOL)


def test_prototype_equals_class_mean():
    """prototype(d) == g + Δ_d == mean(h | d) — the vector `full_rep` injects."""
    features, labels, _, _ = _synthetic()
    axes = extract_concept_vectors(features, labels, classes=CLASSES)

    for c in CLASSES:
        expected = features[labels == c].astype(np.float32).mean(axis=0)
        np.testing.assert_allclose(axes.prototype(c), expected, atol=TOL)


def test_classes_default_to_sorted_unique():
    """Default class order is sorted(unique(labels)), not data order (axes.py:63-65)."""
    features, labels, _, _ = _synthetic()
    axes = extract_concept_vectors(features, labels)
    assert axes.classes == tuple(sorted(CLASSES))


def test_n_samples_counts_each_class():
    features, labels, _, _ = _synthetic(pairs=6)
    axes = extract_concept_vectors(features, labels, classes=CLASSES)
    assert axes.n_samples == {c: 12 for c in CLASSES}


def test_raises_on_absent_class():
    """A pinned class with no samples is an error, not a silent NaN mean."""
    features, labels, _, _ = _synthetic()
    with pytest.raises(ValueError, match="no samples"):
        extract_concept_vectors(features, labels, classes=(*CLASSES, "Diagonal"))


def test_raises_on_label_length_mismatch():
    """Misaligned labels would silently mis-assign every sample."""
    features, labels, _, _ = _synthetic()
    with pytest.raises(ValueError, match="length"):
        extract_concept_vectors(features, labels[:-1])


def test_raises_on_1d_features():
    with pytest.raises(ValueError, match="at least 2-D"):
        extract_concept_vectors(np.arange(10.0), ["Up"] * 10)


def test_concept_axes_by_layer_rejects_non_3d():
    features, labels, _, _ = _synthetic()
    with pytest.raises(ValueError, match=r"expected \(N, L, D\)"):
        concept_axes_by_layer(features, labels)


def test_at_layer_rejects_unstacked_axes():
    """Slicing flat (D,) axes is a layer-mixing bug; it must raise (types.py:52-53)."""
    features, labels, _, _ = _synthetic()
    axes = extract_concept_vectors(features, labels, classes=CLASSES)
    with pytest.raises(ValueError, match="not layer-stacked"):
        axes.at_layer(0)


def test_fp16_input_is_accumulated_in_float32():
    """fp16 features are cast before averaging (axes.py:93).

    Features are stored fp16 on disk; averaging thousands of them in fp16 loses the
    signal to accumulation error. The dtype promotion is a correctness requirement,
    so assert it rather than trusting the default.
    """
    features, labels, _, deltas = _synthetic(dim=8, pairs=64)
    axes = extract_concept_vectors(features.astype(np.float16), labels, classes=CLASSES)

    assert axes.g.dtype == np.float32
    for c in CLASSES:
        # fp16 storage costs ~3 decimal digits; the delta must still be recognisable.
        np.testing.assert_allclose(axes.delta[c], deltas[c], atol=2e-2)


def test_result_is_a_frozen_conceptaxes():
    """ConceptAxes is frozen: axes are shared across scripts and must not be patched."""
    features, labels, _, _ = _synthetic()
    axes = extract_concept_vectors(features, labels, classes=CLASSES)
    assert isinstance(axes, ConceptAxes)
    with pytest.raises(Exception):
        axes.g = np.zeros(axes.dim)  # type: ignore[misc]


def test_mean_magnitude_averages_over_classes():
    """mean_magnitude() is the headline 'direction signal strength' (types.py:63-69)."""
    features, labels, _, _ = _synthetic()
    axes = extract_concept_vectors(features, labels, classes=CLASSES)
    expected = float(np.mean([axes.mag[c] for c in CLASSES]))
    assert axes.mean_magnitude() == pytest.approx(expected, abs=TOL)
