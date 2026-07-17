"""The ConceptAxes container: g, Δ_d, Δ̂_d, ‖Δ_d‖ for one stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np

__all__ = ["ConceptAxes"]


@dataclass(frozen=True)
class ConceptAxes:
    """Direction concept vectors for one stage (one layer, or all layers stacked).

    Replaces the loose `dict(g=..., h_avg_d=..., Delta=..., Delta_hat=..., mag=...)`
    that `mechanism_diagnosis.compute_stats` returned and that every downstream script
    unpacked by string key.

    Attributes:
        g: global mean, shape (...,D). The origin that Δ is measured from.
        delta: class -> Δ_d = mean(h|d) − g, RAW (carries the magnitude).
        delta_hat: class -> unit axis Δ̂_d.
        mag: class -> ‖Δ_d‖. Shape (L,) for layer-stacked input, else scalar.
        classes: pinned class order.
        n_samples: class -> sample count that went into the mean. Small counts make Δ
            noisy; the project's factorial conditions use thousands per direction.
    """

    g: np.ndarray
    delta: Mapping[Hashable, np.ndarray]
    delta_hat: Mapping[Hashable, np.ndarray]
    mag: Mapping[Hashable, np.ndarray]
    classes: tuple[Hashable, ...]
    n_samples: Mapping[Hashable, int]

    @property
    def dim(self) -> int:
        """Feature dimensionality (last axis)."""
        return int(self.g.shape[-1])

    def prototype(self, cls: Hashable) -> np.ndarray:
        """Reconstruct mean(h | class) = g + Δ_class.

        This is the `h_avg_d[d]` that the `full_rep` intervention injects.
        """
        return self.g + self.delta[cls]

    def at_layer(self, layer: int) -> "ConceptAxes":
        """Slice a layer out of layer-stacked axes ((L, D) -> (D,))."""
        if self.g.ndim < 2:
            raise ValueError("axes are not layer-stacked; nothing to slice")
        return ConceptAxes(
            g=self.g[layer],
            delta={d: v[layer] for d, v in self.delta.items()},
            delta_hat={d: v[layer] for d, v in self.delta_hat.items()},
            mag={d: v[layer] for d, v in self.mag.items()},
            classes=self.classes,
            n_samples=self.n_samples,
        )

    def mean_magnitude(self) -> float:
        """‖Δ‖ averaged over classes — the headline 'direction signal strength'.

        This is the quantity behind the in-domain vs OOD magnitude gap
        (shape_color ≈ 48 vs obj_place ≈ 28 at L21 for the Baseline model).
        """
        return float(np.mean([np.asarray(self.mag[d]) for d in self.classes]))
