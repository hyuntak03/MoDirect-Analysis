"""Linear probing: what is linearly readable from a cached representation.

    from modirect.probing import DIRECTION, LETTER, train_linear_probe, get_preset

    res = train_linear_probe(feats, labels, get_preset("legacy_letter"))
    res.accuracy       # a FRACTION; res.accuracy_pct for the published unit

Two targets, and the distinction is the point (`targets`): DIRECTION is what the model
SEES, LETTER is what it must SAY. They diverge — direction stays ~92% at L21 on
obj_place while letter sits at 77% — and that gap is the binding phenomenon. Probing
direction alone will tell you the model is fine when it is not.

`linear` keeps the legacy hyperparameters explicit (`LEGACY_VISION`, `LEGACY_ANSWER`,
`LEGACY_LETTER`) so previously published numbers stay reproducible, with `CANONICAL` for
new work. Preset names are exactly those four — `get_preset("answer")` raises. Read
`linear.py`'s docstring before comparing numbers ACROSS presets: the legacy variants
disagree on test_ratio, epochs, split strategy and std guard, which is why the published
vision and answer-token tables were never strictly comparable.

numpy-only at import time; torch is optional and referenced under TYPE_CHECKING
(`linear.py:58-59`) then imported inside `train_linear_probe` (`:306`), so this package
imports on a host with no model runtime.
"""

from __future__ import annotations

from .linear import (
    CANONICAL,
    LEGACY_ANSWER,
    LEGACY_LETTER,
    LEGACY_VISION,
    PRESETS,
    ProbeConfig,
    ProbeResult,
    get_preset,
    train_linear_probe,
)
from .targets import (
    DIRECTION,
    DIRECTIONS,
    LETTER,
    LETTERS,
    ProbeTarget,
    encode_labels,
    join_letter_labels,
    load_letter_labels,
    resolve_answer_text,
)

__all__ = [
    # linear
    "ProbeConfig",
    "ProbeResult",
    "train_linear_probe",
    "LEGACY_VISION",
    "LEGACY_ANSWER",
    "LEGACY_LETTER",
    "CANONICAL",
    "PRESETS",
    "get_preset",
    # targets
    "DIRECTIONS",
    "LETTERS",
    "ProbeTarget",
    "DIRECTION",
    "LETTER",
    "resolve_answer_text",
    "encode_labels",
    "load_letter_labels",
    "join_letter_labels",
]
