"""Disk I/O for cached features and concept vectors.

    from modirect.io import load_features, load_concept_axes

    feats = load_features("baseline", "answer_token", "obj_place", layer=21)
    axes  = load_concept_axes("baseline", "obj_place")   # committed assets/*.pt

`feature_store` reads the extracted arrays (mmap by default — an answer-token stage is
28 layers of (N, 3584) and does not want to be resident). `concept_store` handles the
eight `assets/concept_vectors/*.pt` files that are committed to the repo on purpose.

This package is importable **without torch**: `concept_store` defers `import torch` into
the only two functions that touch `.pt` files (`concept_store.py:168 load_avg_hidden`,
`:327 save_concept_axes`), so the numpy paths work on a host with no model runtime. Keep it
that way — a module-level torch import here would break `import modirect.io` everywhere,
including under test. The Δ-derivation itself (`avg_hidden_to_axes`) is numpy-only and
therefore testable with no runtime present.

Note `load_concept_axes` returns `(axes, layers)`, not a bare `ConceptAxes`: the committed
files cover L15..L21 and `ConceptAxes.at_layer` indexes POSITIONALLY, so you need `layers`
to address a decoder layer — `axes.at_layer(layers.index(21))`.
"""

from __future__ import annotations

from .concept_store import (
    AVG_HIDDEN_DIRECTIONS,
    AVG_HIDDEN_LAYERS,
    avg_hidden_to_axes,
    list_concept_vectors,
    load_avg_hidden,
    load_concept_axes,
    save_concept_axes,
)
from .feature_store import (
    FeatureRef,
    available_layers,
    load_features,
    load_labels,
    load_meta,
    qids_for,
)

__all__ = [
    # feature_store
    "FeatureRef",
    "load_features",
    "load_labels",
    "load_meta",
    "qids_for",
    "available_layers",
    # concept_store
    "AVG_HIDDEN_LAYERS",
    "AVG_HIDDEN_DIRECTIONS",
    "load_avg_hidden",
    "avg_hidden_to_axes",
    "load_concept_axes",
    "save_concept_axes",
    "list_concept_vectors",
]
