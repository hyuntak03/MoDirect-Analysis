"""Direction concept vectors: Δ_d = mean(h | d) − mean(h | all).

Re-exports the single definition of the project's central quantity, which was
previously re-derived in ten separate scripts (see `axes.py` for the full list).

numpy-only: safe to import without torch or llava.
"""

from __future__ import annotations

from .axes import concept_axes_by_layer, extract_concept_vectors
from .types import ConceptAxes

__all__ = ["ConceptAxes", "concept_axes_by_layer", "extract_concept_vectors"]
