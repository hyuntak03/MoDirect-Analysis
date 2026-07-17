"""Forward hooks that apply direction interventions to the last token.

THE PROTOCOL — this is the part that silently costs you a week if you get it wrong:

    module          hook may WRITE?   why
    ------------    ---------------   ----------------------------------------------
    decoder layer   YES               its return value is threaded into the next layer
    self_attn       NO  (read only)   Qwen2 IGNORES the return value of a self_attn
                                      forward hook. Writing here changes nothing and
                                      produces a silent no-op, not an error.
    mlp             NO  (read only)   same
    mm_projector    YES               vision-side interventions attach here

`extract_attn_mlp_contrib.py` therefore only *reads* from self_attn/mlp, and every
intervention writes at the decoder layer. Keep it that way.

Two further invariants, learned from bugs in the original scripts:

  * ALWAYS clone before writing. `l19_intervention.py`'s hook mutated `output[0]`
    in place, which corrupts the tensor other hooks/caches may still hold.
  * PRESERVE the tuple shape. A decoder layer returns `(hidden, *rest)`; returning a
    bare tensor drops the cache and attention outputs. `12_three_condition_steering.py`
    silently returned the output unmodified in its non-tuple branch, which meant the
    intervention appeared to run while doing nothing.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["last_token_hook", "LastTokenIntervention"]


def _unpack(output: Any):
    """Split a module output into (hidden, rebuild_fn) preserving its container type."""
    if isinstance(output, tuple):
        return output[0], lambda h: (h,) + output[1:]
    return output, lambda h: h


def last_token_hook(fn: Callable[[Any], Any]) -> Callable:
    """Build a forward hook that replaces the LAST token's hidden state with fn(h_last).

    Args:
        fn: maps the (B, D) last-token slice to a new (B, D) slice.

    Returns:
        A `register_forward_hook`-compatible callable.

    The hook clones the hidden state before writing and rebuilds the original container
    (tuple vs bare tensor), so it is safe to attach to a Qwen2 decoder layer and to
    Qwen3-VL's (which returns a bare Tensor).
    """

    def hook(module, inputs, output):
        hidden, rebuild = _unpack(output)
        new_last = fn(hidden[:, -1, :])
        hidden = hidden.clone()
        hidden[:, -1, :] = new_last.to(hidden.dtype)
        return rebuild(hidden)

    return hook


class LastTokenIntervention:
    """Context manager applying a named condition at one decoder layer.

    Usage:

        axes_L = axes.at_layer(21)
        with LastTokenIntervention(model, 21, axes_L, "clean", direction,
                                   magnitude=2 * mag_sc):
            logits = model(inputs_embeds=..., ...).logits

    The hook is removed on exit even if the forward raises, which the original scripts
    handled with an explicit try/finally around every call.
    """

    def __init__(self, model, layer: int, axes, condition: str, cls,
                 *, magnitude: float | None = None, factor: float = 2.0):
        from .operators import apply_condition

        self._layers = _decoder_layers(model)
        self._layer = layer
        self._handle = None
        self._noop = condition == "no_swap"

        import torch

        def _fn(h_last):
            g = torch.as_tensor(axes.g)
            dh = torch.as_tensor(axes.delta_hat[cls])
            proto = torch.as_tensor(axes.prototype(cls)) if condition == "full_rep" else None

            class _Shim:  # minimal ConceptAxes view with tensors already materialised
                g = None
                delta_hat = None

                def prototype(self, _):
                    return proto

            shim = _Shim()
            shim.g = g
            shim.delta_hat = {cls: dh}
            return apply_condition(condition, h_last.float(), shim, cls,
                                   magnitude=magnitude, factor=factor)

        self._fn = _fn

    def __enter__(self):
        if not self._noop:
            self._handle = self._layers[self._layer].register_forward_hook(
                last_token_hook(self._fn))
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def _decoder_layers(model):
    """Locate the decoder layer list across the LLaVA / Qwen wrappers used here."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    raise AttributeError(f"cannot locate decoder layers on {type(model).__name__}")
