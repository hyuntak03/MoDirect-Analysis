"""Loading and saving ConceptAxes — including the 8 committed `.pt` concept vectors.

THE COMMITTED ASSETS
--------------------
`assets/concept_vectors/{model}_{task}.pt`, 8 files = 2 models x 4 tasks:

    vanilla_{shape_color,obj_color,shape_place,obj_place}.pt
    baseline_{shape_color,obj_color,shape_place,obj_place}.pt

There is no `delta_*` — `02_extract_avg_hidden.py:124` only accepts
`choices=["vanilla", "baseline"]`, so the Delta model's averaged hiddens were never
extracted by this path.

EXACT SAVED LAYOUT — verified by unpickling the committed files, not just read off the
script (`02_extract_avg_hidden.py:138` saves the dict):

    {
      "avg":    {direction: {layer: Tensor(3584,) float32}},   # 4 dirs x 7 layers = 28
      "counts": {direction: int},                              # 100 each, balanced
      "layers": [15, 16, 17, 18, 19, 20, 21],
    }

    directions   "up", "right", "down", "left"      (:37, insertion order)
    layers       L15..L21 inclusive                 (:38, `range(15, 22)`)
    dtype        float32                            (:110 casts before the CPU copy)
    D            3584                               (Qwen2-7B hidden size)

The layer indexing convention is the one that bites: `avg[d][L]` is keyed by DECODER LAYER
L, read from `output.hidden_states[L + 1]` (:109-110). `hidden_states[0]` is the embedding
output, so `hidden_states[L+1]` is the output of decoder layer L. Layer keys here are
directly comparable to the L21 in "L21 canonical axis" and to
`modirect.interventions.hooks.LastTokenIntervention(model, 21, ...)`, which registers on
`model.layers[21]`. No off-by-one correction is needed on load — it was already applied at
extraction.

WHY `delta` MUST BE DERIVED ON LOAD
-----------------------------------
`02_extract_avg_hidden` stores per-direction means h_avg(d) ONLY. It never computes a
global mean and never subtracts one (`:118` is a bare `sums[d][l] / counts[d]`). The saved
tensors are therefore PROTOTYPES, not concept vectors: they sit at g + Δ_d, and ‖h_avg(d)‖
is dominated by g — the residual-stream component common to every sample, which carries no
direction information at all.

So this loader reconstructs the origin before handing back a `ConceptAxes`:

    g   = Σ_d counts[d] · h_avg(d) / Σ_d counts[d]
    Δ_d = h_avg(d) − g

The count weighting recovers the TRUE global mean exactly, because each h_avg(d) is a
complete mean over its own group and the groups partition the sample set. It is weighted
rather than a plain `mean()` over the four vectors as a matter of correctness under
imbalance; for the committed files counts are 100/100/100/100, so the two agree and the
loaded Δ is exact.

Consequence worth stating plainly: `‖Δ_d‖` from these files is NOT comparable to the
magnitudes in CLAUDE.md section O (SC ≈ 48, OP ≈ 28 at L21). Those were measured on the
factorial dataset over thousands of samples per direction; these are R2R canonical samples
at n=100 per direction. Same definition, different population.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from ..concepts.types import ConceptAxes
from ..config.directions import DIRS_FACTORIAL_CLOCKWISE

__all__ = [
    "AVG_HIDDEN_LAYERS",
    "AVG_HIDDEN_DIRECTIONS",
    "CONCEPT_VECTOR_RELPATH",
    "concept_vector_path",
    "load_avg_hidden",
    "avg_hidden_to_axes",
    "load_concept_axes",
    "save_concept_axes",
    "list_concept_vectors",
]

_EPS = 1e-9

#: Where the committed `.pt` files live, relative to `repo_root()`. They are checked into
#: the repo (~400 KB each) rather than regenerated, because reproducing them needs the GPU
#: runtime and the R2R canonical videos — neither of which is present on an analysis host.
CONCEPT_VECTOR_RELPATH = "assets/concept_vectors"


def concept_vector_path(
    model: str, task: str, *, root: str | os.PathLike | None = None
) -> Path:
    """Resolve `{root}/{model}_{task}.pt`.

    Args:
        model: "vanilla" or "baseline". There is no "delta": `02_extract_avg_hidden.py:124`
            declares `choices=["vanilla", "baseline"]`, so the Delta model's averaged
            hiddens were never extracted by this path.
        task: short task name, e.g. "obj_place" — NOT the long lmms-eval task name the
            feature cache uses. The filenames were built from `--task`, whose choices are
            the four short names (`:125`).
        root: defaults to `repo_root()/assets/concept_vectors`.

    Returns:
        The path. Existence is not checked; `load_concept_axes` reports that.
    """
    if root is None:
        from ..config.paths import repo_root

        root = repo_root() / CONCEPT_VECTOR_RELPATH
    return Path(root) / f"{model}_{task}.pt"

#: Layer range the committed files cover: `range(15, 22)` (`02_extract_avg_hidden.py:38`).
#: L15..L21 was chosen to bracket the letter-binding transition (L15->L16) and the
#: canonical-axis arrival (L20-L21). Nothing outside this window is on disk.
AVG_HIDDEN_LAYERS: tuple[int, ...] = tuple(range(15, 22))

#: Direction keys, in the file's insertion order — `DIRS` at `02_extract_avg_hidden.py:37`,
#: confirmed by unpickling: up, right, down, left. That is Ordering 2
#: (`modirect.config.directions.DIRS_FACTORIAL_CLOCKWISE`), NOT the Ordering 1 that
#: `labels.npy` integers follow. It is safe here only because these are dict STRING keys,
#: never indices — `avg["up"]`, not `avg[3]`. Never zip this against a `labels.npy` column;
#: the two orderings disagree at every position. See `modirect.config.directions`.
AVG_HIDDEN_DIRECTIONS: tuple[str, ...] = DIRS_FACTORIAL_CLOCKWISE


def list_concept_vectors(root: str | os.PathLike) -> list[tuple[str, str]]:
    """List (model, task) pairs available under a concept-vector directory.

    Splits `{model}_{task}.pt` on the FIRST underscore, since every task name itself
    contains one (`shape_color`, `obj_place`, ...) while the model names here do not
    (`vanilla`, `baseline`).

    Returns:
        Sorted (model, task) pairs, e.g. `[("baseline", "obj_color"), ...]`.
    """
    pairs = []
    for p in sorted(Path(root).glob("*.pt")):
        model, _, task = p.stem.partition("_")
        if task:
            pairs.append((model, task))
    return pairs


def load_avg_hidden(path: str | os.PathLike) -> dict[str, Any]:
    """Load a raw `{model}_{task}.pt` exactly as `02_extract_avg_hidden.py:138` wrote it.

    Prefer `load_concept_axes`, which returns a `ConceptAxes` with Δ already derived. This
    is the escape hatch for inspecting the prototypes directly.

    Args:
        path: the `.pt` file.

    Returns:
        `{"avg": {dir: {layer: Tensor}}, "counts": {dir: int}, "layers": [15..21]}` — with
        torch Tensors, as saved.

    Raises:
        FileNotFoundError: if absent.
        ImportError: torch is required to unpickle these; it is imported lazily so that
            importing `modirect.io` stays free on hosts without torch.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no concept vector file at {path}")
    import torch

    # weights_only=True: these hold plain tensors/ints/lists, so the safe loader suffices,
    # and it is the default from torch 2.6 onward anyway.
    return torch.load(path, map_location="cpu", weights_only=True)


def avg_hidden_to_axes(
    avg: Mapping[Hashable, Mapping[int, Any]],
    counts: Mapping[Hashable, int],
    layers: Sequence[int],
    *,
    classes: Sequence[Hashable] | None = None,
) -> ConceptAxes:
    """Turn saved per-direction prototypes into layer-stacked ConceptAxes.

    Derives the origin the file never stored (see the module docstring):
    `g = Σ_d counts[d]·h_avg(d) / Σ_d counts[d]`, then `Δ_d = h_avg(d) − g`.

    Args:
        avg: `{direction: {layer: (D,) tensor or array}}`.
        counts: `{direction: n}`, used to weight the origin.
        layers: layer keys to stack, in the order they will occupy the L axis.
        classes: direction order. Defaults to the order of `avg`, which for a file written
            by `02_extract_avg_hidden` is `AVG_HIDDEN_DIRECTIONS`. Pin it when comparing
            axes across files.

    Returns:
        ConceptAxes with `g` of shape (L, D) and each `delta[d]` of shape (L, D) — the
        layer-stacked form `ConceptAxes.at_layer` slices. `mag[d]` is (L,).

        The returned `layers` ordering is positional: `axes.at_layer(0)` is `layers[0]`
        (= L15 for the committed files), NOT decoder layer 0. Use
        `layer_index = layers.index(21)` to address L21. This is the one place where the
        project's layer numbering does not survive into the array, because `ConceptAxes`
        indexes its stack positionally.

    Raises:
        ValueError: if a class is absent from `avg` or `counts`, is missing a layer, or if
            the counts sum to zero.
    """
    if classes is None:
        classes = tuple(avg.keys())
    classes = tuple(classes)

    # Membership is checked against BOTH mappings up front. `counts` is consulted first
    # when summing, so without this an unknown class surfaced as a bare KeyError from the
    # weighting expression rather than as the named error this function documents.
    for d in classes:
        if d not in avg:
            raise ValueError(f"direction {d!r} absent from avg; have {sorted(avg)}")
        if d not in counts:
            raise ValueError(f"direction {d!r} absent from counts; have {sorted(counts)}")

    total = sum(int(counts[d]) for d in classes)
    if total <= 0:
        raise ValueError(f"counts sum to {total}; nothing to average")

    stacked: dict[Hashable, np.ndarray] = {}
    for d in classes:
        try:
            rows = [np.asarray(avg[d][l], dtype=np.float32) for l in layers]
        except KeyError as exc:
            raise ValueError(
                f"direction {d!r} has no layer {exc.args[0]!r}; have {sorted(avg[d])}"
            ) from None
        stacked[d] = np.stack(rows, axis=0)  # (L, D)

    # Count-weighted origin == the true global mean, since the per-direction means are
    # complete and the directions partition the samples.
    g = sum(int(counts[d]) * stacked[d] for d in classes) / float(total)

    delta = {d: stacked[d] - g for d in classes}
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
        classes=classes,
        n_samples={d: int(counts[d]) for d in classes},
    )


def load_concept_axes(
    model: str,
    task: str,
    *,
    root: str | os.PathLike | None = None,
    path: str | os.PathLike | None = None,
    classes: Sequence[Hashable] | None = None,
) -> tuple[ConceptAxes, tuple[int, ...]]:
    """Load a committed `.pt` and return layer-stacked ConceptAxes.

    Derives Δ on load — the file stores prototypes only. See the module docstring.

    Args:
        model: "vanilla" or "baseline".
        task: short task name, e.g. "obj_place".
        root: directory holding the `.pt` files. Defaults to the committed
            `assets/concept_vectors`.
        path: an explicit file, bypassing `model`/`task` resolution.
        classes: direction order to pin. Defaults to the file's key order (Ordering 2).

    Returns:
        `(axes, layers)`. `layers` gives the DECODER-LAYER number of each stack position —
        `(15, ..., 21)` for the committed files — and you need it to address a layer,
        because `ConceptAxes.at_layer` is POSITIONAL:

            axes, layers = load_concept_axes("baseline", "obj_place")
            l21 = axes.at_layer(layers.index(21))   # not at_layer(21)

    Raises:
        FileNotFoundError: if the file is absent.

    Example:
        >>> axes, layers = load_concept_axes("baseline", "obj_place")  # doctest: +SKIP
        >>> layers                                                     # doctest: +SKIP
        (15, 16, 17, 18, 19, 20, 21)
    """
    if path is None:
        path = concept_vector_path(model, task, root=root)
    blob = load_avg_hidden(path)
    layers = tuple(int(l) for l in blob["layers"])
    axes = avg_hidden_to_axes(blob["avg"], blob["counts"], layers, classes=classes)
    return axes, layers


def save_concept_axes(
    axes: ConceptAxes,
    path: str | os.PathLike,
    *,
    layers: Sequence[int] | None = None,
) -> Path:
    """Save ConceptAxes back in the `02_extract_avg_hidden` layout, as `.pt`.

    Writes PROTOTYPES (g + Δ_d), not Δ, so the file stays byte-compatible with the
    committed assets and with any legacy script that reads `blob["avg"][d][l]` directly
    (e.g. `03_swap_*.py`). Round-tripping through `load_concept_axes` reconstructs the same
    Δ, since the origin is recoverable from the count-weighted prototypes.

    Args:
        axes: layer-stacked (g is (L, D)) or single-layer (g is (D,)).
        path: destination `.pt`.
        layers: decoder-layer number for each stack position. Required for layer-stacked
            axes — the stack alone does not know it starts at 15. For single-layer axes
            pass a length-1 sequence.

    Returns:
        The written path.

    Raises:
        ValueError: if `layers` is missing or its length disagrees with the stack depth.
    """
    import torch

    path = Path(path)
    stacked = axes.g.ndim >= 2
    depth = int(axes.g.shape[0]) if stacked else 1

    if layers is None:
        raise ValueError(
            "pass `layers`: the decoder-layer number of each stack position is not "
            "recoverable from the array (the committed files start at L15, not L0)"
        )
    layers = tuple(int(l) for l in layers)
    if len(layers) != depth:
        raise ValueError(f"{len(layers)} layers given but stack depth is {depth}")

    avg: dict[Any, dict[int, Any]] = {}
    for d in axes.classes:
        proto = np.asarray(axes.prototype(d), dtype=np.float32)
        rows = proto if stacked else proto[None, :]
        avg[d] = {l: torch.from_numpy(np.ascontiguousarray(rows[i])) for i, l in enumerate(layers)}

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"avg": avg, "counts": dict(axes.n_samples), "layers": list(layers)}, path
    )
    return path
