"""Tests for `modirect.interventions.operators` — the magnitude conditions.

Each operator is asserted against its algebraic contract on the on-axis projection
    proj(h) = ⟨h − g, Δ̂_d⟩
and, where it matters, on the off-axis residual
    off(h) = (h − g) − proj(h)·Δ̂_d.

The on/off split is the whole point of the ablation table (operators.py:19-27): the
conditions differ precisely in what they do to each half, and the paper's claim
(`clean(m = 2×mag_SC)` reproduces `full_rep`) only means something if `clean` sets the
on-axis component to an exact absolute target while leaving the off-axis alone.

numpy only — no torch. The operators are framework-agnostic (operators.py:3-5), so the
numpy path under test is the same code the live hooks run.
"""

from __future__ import annotations

import numpy as np
import pytest

from modirect.concepts import extract_concept_vectors
from modirect.interventions.operators import (
    CONDITIONS,
    _dot_last,
    add_canon,
    amp,
    apply_condition,
    clean,
    on_axis,
    project_on_axis,
    remove_own,
)

# float32 axes vs float64 hidden states; 1e-4 absolute is ~6 orders below the
# magnitudes under test (‖Δ‖ ≈ 28–96 in the real L21 data).
TOL = 1e-4

CLASSES = ("Up", "Down", "Left", "Right")
CLS = "Up"


@pytest.fixture()
def axes():
    """ConceptAxes over synthetic (N, D) data — a stand-in for one layer's last token."""
    rng = np.random.default_rng(7)
    dim = 24
    base = rng.normal(size=dim) * 2.0
    raw = rng.normal(size=(len(CLASSES), dim))
    raw -= raw.mean(axis=0, keepdims=True)

    feats, labels = [], []
    for i, c in enumerate(CLASSES):
        noise = rng.normal(size=(8, dim))
        feats.append(np.concatenate([base + raw[i] + noise, base + raw[i] - noise]))
        labels.extend([c] * 16)
    return extract_concept_vectors(np.concatenate(feats), np.array(labels),
                                   classes=CLASSES)


@pytest.fixture()
def h(axes):
    """A single (D,) hidden state that is NOT on the axis — off-axis content present."""
    rng = np.random.default_rng(11)
    return (axes.prototype(CLS) + rng.normal(size=axes.dim) * 1.5).astype(np.float64)


def _proj(h, axes, cls=CLS) -> float:
    return float(np.ravel(project_on_axis(h, axes.g, axes.delta_hat[cls]))[0])


def _off(h, axes, cls=CLS) -> np.ndarray:
    """The off-axis residual (h − g) − proj·Δ̂."""
    dh = axes.delta_hat[cls]
    return (h - axes.g) - _proj(h, axes, cls) * dh


# ------------------------------------------------------------------ project_on_axis

def test_projection_is_centred_on_g(axes):
    """proj is measured from g, not from the origin (operators.py:72-75).

    Without centring, the projection is dominated by the sample-independent component
    of the residual stream, which is large and carries no direction information — so
    h = g must project to exactly 0.
    """
    assert _proj(axes.g.astype(np.float64), axes) == pytest.approx(0.0, abs=TOL)


def test_projection_of_prototype_is_the_magnitude(axes):
    """proj(g + Δ_d) == ‖Δ_d‖ — the prototype sits at `mag` along its own axis."""
    assert _proj(axes.prototype(CLS), axes) == pytest.approx(
        float(axes.mag[CLS]), abs=TOL)


# ------------------------------------------------------------------ conditions

def test_remove_own_zeroes_the_on_axis_projection(axes, h):
    """remove_own: h − proj·Δ̂ ⇒ proj' == 0.

    The load-bearing control (operators.py:29-31): ablating this axis is what drops
    real accuracy to 52.4%, establishing the model actually reads it.
    """
    out = remove_own(h, axes.g, axes.delta_hat[CLS])
    assert _proj(out, axes) == pytest.approx(0.0, abs=TOL)


def test_remove_own_preserves_the_off_axis_residual(axes, h):
    """Only the on-axis component is removed; everything else survives."""
    out = remove_own(h, axes.g, axes.delta_hat[CLS])
    np.testing.assert_allclose(_off(out, axes), _off(h, axes), atol=TOL)


@pytest.mark.parametrize("magnitude", [0.0, 28.0, 48.0, 96.0])
def test_clean_sets_projection_to_exact_absolute_magnitude(axes, h, magnitude):
    """clean(m): proj' == m exactly, whatever the sample started at.

    `magnitude` is an ABSOLUTE target, not a scale factor (operators.py:93-94). The
    real values swept are 14/28/48/96 (OP_half, OP_mean, SC_mean, 2×SC_mean).
    """
    out = clean(h, axes.g, axes.delta_hat[CLS], magnitude=magnitude)
    assert _proj(out, axes) == pytest.approx(magnitude, abs=TOL)


def test_clean_leaves_off_axis_untouched(axes, h):
    """clean replaces the on-axis component only — identity/letter content stays.

    This is what distinguishes clean from on_axis, and why clean(m=96) matching
    full_rep implies the recovery is magnitude rather than prototype content.
    """
    out = clean(h, axes.g, axes.delta_hat[CLS], magnitude=48.0)
    np.testing.assert_allclose(_off(out, axes), _off(h, axes), atol=TOL)


def test_add_canon_increases_projection_by_exactly_m(axes, h):
    """add_canon: proj' == proj + m — adds signal without removing the sample's own."""
    m = 48.0
    before = _proj(h, axes)
    out = add_canon(h, axes.g, axes.delta_hat[CLS], magnitude=m)
    assert _proj(out, axes) == pytest.approx(before + m, abs=TOL)


def test_add_canon_leaves_off_axis_untouched(axes, h):
    out = add_canon(h, axes.g, axes.delta_hat[CLS], magnitude=48.0)
    np.testing.assert_allclose(_off(out, axes), _off(h, axes), atol=TOL)


def test_amp_2x_doubles_projection_and_preserves_off_axis(axes, h):
    """amp(factor=2): proj' == 2·proj AND off-axis unchanged.

    The off-axis half is the property that separates amp from on_axis: amp scales the
    direction signal *in situ*, on_axis scales nothing and destroys the rest. Both
    raise accuracy, but only amp does so without discarding content.
    """
    before_proj = _proj(h, axes)
    before_off = _off(h, axes)

    out = amp(h, axes.g, axes.delta_hat[CLS], factor=2.0)

    assert _proj(out, axes) == pytest.approx(2.0 * before_proj, abs=TOL)
    np.testing.assert_allclose(_off(out, axes), before_off, atol=TOL)


@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0, 3.0])
def test_amp_scales_projection_by_factor(axes, h, factor):
    """amp is linear in `factor`, and factor=1 is the identity."""
    before = _proj(h, axes)
    out = amp(h, axes.g, axes.delta_hat[CLS], factor=factor)
    assert _proj(out, axes) == pytest.approx(factor * before, abs=TOL)


def test_on_axis_zeroes_the_off_axis_residual(axes, h):
    """on_axis: g + proj·Δ̂ ⇒ off-axis residual == 0, on-axis preserved.

    Coarse by design (operators.py:108-111): it discards identity/letter content too,
    which is why it recovers less than add_canon — evidence some off-axis content is
    useful.
    """
    before_proj = _proj(h, axes)
    out = on_axis(h, axes.g, axes.delta_hat[CLS])

    np.testing.assert_allclose(_off(out, axes), np.zeros(axes.dim), atol=TOL)
    assert _proj(out, axes) == pytest.approx(before_proj, abs=TOL)


def test_full_rep_replaces_with_the_prototype(axes, h):
    """full_rep injects g + Δ_d outright."""
    out = apply_condition("full_rep", h, axes, CLS)
    np.testing.assert_allclose(out, axes.prototype(CLS), atol=TOL)


def test_clean_at_two_times_magnitude_matches_full_rep_projection(axes, h):
    """The paper's central claim, as an algebraic identity.

    `clean(m)` and `full_rep` land on the SAME on-axis projection when m = ‖Δ_d‖ —
    they differ only off-axis. That is exactly why the empirical clean(m=96) ≡
    full_rep (+17.6pp both) says the recovery is carried by magnitude.
    """
    m = float(axes.mag[CLS])
    cleaned = clean(h, axes.g, axes.delta_hat[CLS], magnitude=m)
    replaced = apply_condition("full_rep", h, axes, CLS)

    assert _proj(cleaned, axes) == pytest.approx(_proj(replaced, axes), abs=TOL)


# ------------------------------------------------------------------ apply_condition

def test_no_swap_is_the_identity(axes, h):
    """"no_swap" returns h unchanged so callers can loop over conditions uniformly."""
    out = apply_condition("no_swap", h, axes, CLS)
    np.testing.assert_array_equal(out, h)


def test_apply_condition_dispatches_to_the_same_result_as_the_operator(axes, h):
    """Dispatch must not diverge from calling the operator directly."""
    direct = clean(h, axes.g, axes.delta_hat[CLS], magnitude=48.0)
    viadisp = apply_condition("clean", h, axes, CLS, magnitude=48.0)
    np.testing.assert_allclose(viadisp, direct, atol=TOL)


def test_apply_condition_amp_2x_uses_factor_two(axes, h):
    before = _proj(h, axes)
    out = apply_condition("amp_2x", h, axes, CLS)
    assert _proj(out, axes) == pytest.approx(2.0 * before, abs=TOL)


def test_apply_condition_does_not_mutate_h(axes, h):
    """Operators are pure; the hook clones, but the arithmetic must not write in place."""
    original = h.copy()
    for name in ("amp_2x", "on_axis", "remove_own", "full_rep"):
        apply_condition(name, h, axes, CLS)
    apply_condition("clean", h, axes, CLS, magnitude=48.0)
    np.testing.assert_array_equal(h, original)


def test_apply_condition_raises_on_unknown_name(axes, h):
    with pytest.raises(ValueError, match="unknown condition"):
        apply_condition("amplify_please", h, axes, CLS)


@pytest.mark.parametrize("name", ["clean", "clean_sc", "add_canon"])
def test_apply_condition_requires_magnitude(axes, h, name):
    """clean/add_canon take an absolute ‖Δ‖; defaulting it would silently inject 0."""
    with pytest.raises(ValueError, match="requires `magnitude`"):
        apply_condition(name, h, axes, CLS)


def test_conditions_registry_covers_the_ablation_table():
    """Every named row of the paper's table is dispatchable."""
    for name in ("amp_2x", "clean", "clean_sc", "add_canon", "on_axis",
                 "remove_own", "full_rep"):
        assert name in CONDITIONS


def test_dot_last_works_on_numpy2_style_arrays():
    """Guard the numpy/torch branch discrimination against the numpy 2.0 `.device` attr.

    numpy 1.26 -> 2.0 added `ndarray.device`. `_dot_last` used to treat "has .device" as
    "is a torch tensor", which was true in numpy 1.x and is false from 2.0 on. This host
    runs numpy 1.24 so the bug was dormant; the subclass below exposes the attribute to
    pin the contract regardless of the installed numpy.

    `_dot_last` now branches on `.dim` (torch-only), so this passes; the xfail(strict)
    marker that tracked the bug was dropped when operators.py was fixed.
    """
    class _NumPy2Array(np.ndarray):
        @property
        def device(self):  # what numpy>=2.0 exposes
            return "cpu"

    a = np.arange(6.0).reshape(2, 3).view(_NumPy2Array)
    b = np.ones((2, 3)).view(_NumPy2Array)

    got = _dot_last(a, b)
    np.testing.assert_allclose(np.asarray(got), np.array([[3.0], [12.0]]), atol=TOL)


def test_operators_are_batched_over_leading_axes(axes):
    """(B, D) batches project row-wise — the hook passes a (B, D) last-token slice."""
    rng = np.random.default_rng(3)
    batch = axes.prototype(CLS) + rng.normal(size=(5, axes.dim))

    out = clean(batch, axes.g, axes.delta_hat[CLS], magnitude=48.0)
    proj = project_on_axis(out, axes.g, axes.delta_hat[CLS])

    assert proj.shape == (5, 1)
    np.testing.assert_allclose(proj, np.full((5, 1), 48.0), atol=TOL)
