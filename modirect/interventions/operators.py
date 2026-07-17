"""Magnitude interventions on the direction axis.

Every operator here is a pure function of (h, axes, class) -> h'. They are framework
agnostic: they work on torch tensors and on numpy arrays, so the same code is used by
the live decoder-layer hooks and by offline tests.

The canonical form, from `mechanism_diagnosis.make_hook:115-141`, given the last-token
hidden h, the origin g, and the unit axis Δ̂_d:

    proj = ⟨h − g, Δ̂_d⟩          the sample's own signed on-axis component
    h'   = h − proj·Δ̂_d + m·Δ̂_d   remove own, re-inject magnitude m      ("clean")

That is the `h − v_d + v̂_d·m` shape: `proj·Δ̂_d` is the sample's own direction vector,
and `m·Δ̂_d` is the controlled replacement.

The named conditions below reproduce the paper's ablation table exactly. Reference
numbers, Baseline model / obj_place / L21 / n=500, baseline accuracy 68.80%:

    condition     formula                              acc      Δ
    no_swap       h                                    68.80%   —
    amp_2x        h + proj·Δ̂                           73.80%   +5.0pp
    clean         h − proj·Δ̂ + m·Δ̂   (m = mag_SC)      78.80%   +10.0pp
    add_canon     h + m·Δ̂            (no removal)      80.40%   +11.6pp
    on_axis       g + proj·Δ̂         (off-axis nuked)  78.60%   +9.8pp
    remove_own    h − proj·Δ̂         (control)         52.40%   −16.4pp
    full_rep      g + Δ_d            (prototype)       86.40%   +17.6pp
    clean(m=96)   h − proj·Δ̂ + 96·Δ̂  (2×SC)            86.40%   +17.6pp

`remove_own` is the load-bearing control: it drops accuracy well below baseline (but
stays above the 25% chance floor), which is what establishes that the model is actually
reading this axis rather than the axis being an artefact of the probe.

`clean(m = 2 × mag_SC)` reproducing `full_rep` to the decimal is the paper's central
claim — the recovery is magnitude, not prototype content.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "project_on_axis",
    "apply_condition",
    "CONDITIONS",
    "amp",
    "clean",
    "add_canon",
    "on_axis",
    "remove_own",
    "full_rep",
]


def _as_like(vec: Any, ref: Any) -> Any:
    """Move `vec` onto the device/dtype of `ref` when both are torch tensors."""
    if hasattr(ref, "device") and hasattr(vec, "to"):
        return vec.to(device=ref.device, dtype=ref.dtype)
    return vec


def _dot_last(a: Any, b: Any) -> Any:
    """⟨a, b⟩ over the last axis, keeping dims so it broadcasts back onto a.

    torch and numpy spell the same reduction differently (`sum(dim=, keepdim=)` vs
    `sum(axis, keepdims=)`), so the branch has to identify the framework. It keys on
    `.dim` — a torch-only method — rather than on `.device`: numpy 2.0 added
    `ndarray.device`, so the old `.device` test silently routed every numpy array into
    the torch branch and raised `TypeError: sum() got an unexpected keyword 'dim'`.
    numpy is unpinned (pyproject.toml:29), so any fresh install would hit that.

    The module stays import-free of both frameworks (see module docstring), hence
    duck-typing rather than `isinstance`.
    """
    if hasattr(a, "sum"):
        return ((a * b).sum(dim=-1, keepdim=True) if hasattr(a, "dim")
                else (a * b).sum(-1, keepdims=True))
    raise TypeError(f"unsupported array type: {type(a)}")


def project_on_axis(h: Any, g: Any, delta_hat: Any) -> Any:
    """proj = ⟨h − g, Δ̂⟩ — the sample's own signed component along the axis.

    Centring on `g` matters: without it the projection is dominated by the component of
    the residual stream that is common to every sample, which is large and carries no
    direction information.
    """
    g = _as_like(g, h)
    delta_hat = _as_like(delta_hat, h)
    return _dot_last(h - g, delta_hat)


# --------------------------------------------------------------------- conditions

def amp(h, g, delta_hat, *, factor: float = 2.0, **_):
    """Scale the sample's own on-axis component by `factor` (default 2× => `amp_2x`)."""
    d = _as_like(delta_hat, h)
    proj = project_on_axis(h, g, delta_hat)
    return h + (factor - 1.0) * proj * d


def clean(h, g, delta_hat, *, magnitude: float, **_):
    """h − proj·Δ̂ + magnitude·Δ̂ — remove own on-axis, re-inject a controlled one.

    The workhorse. `magnitude` is an ABSOLUTE target, not a scale factor: pass the
    in-domain mean ‖Δ‖ for `clean_sc`, or 2× it for the full-recovery result.
    """
    d = _as_like(delta_hat, h)
    proj = project_on_axis(h, g, delta_hat)
    return h - proj * d + magnitude * d


def add_canon(h, g, delta_hat, *, magnitude: float, **_):
    """h + magnitude·Δ̂ — add signal without removing the sample's own."""
    return h + magnitude * _as_like(delta_hat, h)


def on_axis(h, g, delta_hat, **_):
    """g + proj·Δ̂ — keep only the on-axis component; destroy everything else.

    Note this discards identity/letter content too, so it is a coarse instrument: it
    recovers less than `add_canon`, which is how we know some off-axis content is useful.
    """
    g_l = _as_like(g, h)
    d = _as_like(delta_hat, h)
    proj = project_on_axis(h, g, delta_hat)
    return g_l + proj * d


def remove_own(h, g, delta_hat, **_):
    """h − proj·Δ̂ — ablate the direction signal. CONTROL: expect accuracy to fall."""
    d = _as_like(delta_hat, h)
    proj = project_on_axis(h, g, delta_hat)
    return h - proj * d


def full_rep(h, g, delta_hat, *, prototype, **_):
    """Replace the token outright with the class prototype g + Δ_d."""
    proto = _as_like(prototype, h)
    return proto.expand_as(h) if hasattr(proto, "expand_as") and proto.ndim < h.ndim else (
        h * 0 + proto)


CONDITIONS = {
    "amp": amp,
    "amp_2x": lambda h, g, dh, **kw: amp(h, g, dh, factor=2.0),
    "clean": clean,
    "clean_sc": clean,
    "add_canon": add_canon,
    "on_axis": on_axis,
    "remove_own": remove_own,
    "full_rep": full_rep,
}


def apply_condition(name: str, h, axes, cls, *, magnitude: float | None = None,
                    factor: float = 2.0):
    """Apply a named condition to `h` using `axes` (a ConceptAxes) for class `cls`.

    Args:
        name: key of `CONDITIONS`. "no_swap" is accepted and returns h unchanged, so
            callers can loop over conditions uniformly including the baseline.
        h: (..., D) last-token hidden state. Not mutated.
        axes: ConceptAxes for the SAME layer h came from. Mixing layers is the classic
            error here — the L14 axis is near-orthogonal to the L21 axis (cos ≈ 0.04),
            so injecting along the wrong one is a no-op at best.
        cls: the class (direction) whose axis to act on.
        magnitude: absolute target ‖Δ‖ for `clean`/`add_canon`. Required for those.

    Returns:
        The modified hidden state, same type/shape as `h`.
    """
    if name == "no_swap":
        return h
    if name not in CONDITIONS:
        raise ValueError(f"unknown condition {name!r}; have {sorted(CONDITIONS) + ['no_swap']}")

    fn = CONDITIONS[name]
    kwargs: dict[str, Any] = {"factor": factor}
    if name in ("clean", "clean_sc", "add_canon"):
        if magnitude is None:
            raise ValueError(f"condition {name!r} requires `magnitude` (an absolute ‖Δ‖)")
        kwargs["magnitude"] = magnitude
    if name == "full_rep":
        kwargs["prototype"] = axes.prototype(cls)

    return fn(h, axes.g, axes.delta_hat[cls], **kwargs)
