"""Input features of the playground: raw columns plus the Playground's derived features."""

from __future__ import annotations

import numpy as np

# Derived features are always computed from the first two base columns ("x1" and "x2").
DERIVED_KEYS = ("sq1", "sq2", "prod", "sin1", "sin2")


def column_key(index: int) -> str:
    return f"col:{index}"


class FeatureMap:
    """Turns raw base columns into the scaled feature matrix the network sees.

    ``selected`` holds feature keys: ``"col:<i>"`` for base column *i*, or one of
    :data:`DERIVED_KEYS`. The map is fitted on the training rows so that every feature is
    rescaled to [-1, 1]; new data is then transformed with the same scaling.
    """

    def __init__(self, base_names: list[str], selected: list[str]) -> None:
        self.base_names = list(base_names)
        self.selected = [k for k in selected if self.is_available(k)]
        self.lo = np.zeros(len(self.selected))
        self.span = np.ones(len(self.selected))

    # ----- keys -------------------------------------------------------------------------------

    def is_available(self, key: str) -> bool:
        n = len(self.base_names)
        if key.startswith("col:"):
            return 0 <= int(key[4:]) < n
        if key in ("sq1", "sin1"):
            return n >= 1
        if key in ("sq2", "sin2", "prod"):
            return n >= 2
        return False

    def label(self, key: str) -> str:
        b = self.base_names
        if key.startswith("col:"):
            return b[int(key[4:])]
        return {
            "sq1": f"{b[0]}²",
            "sq2": f"{b[1]}²" if len(b) > 1 else "",
            "prod": f"{b[0]}·{b[1]}" if len(b) > 1 else "",
            "sin1": f"sin({b[0]})",
            "sin2": f"sin({b[1]})" if len(b) > 1 else "",
        }[key]

    @property
    def labels(self) -> list[str]:
        return [self.label(k) for k in self.selected]

    # ----- transform --------------------------------------------------------------------------

    def _raw(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        cols = []
        for key in self.selected:
            if key.startswith("col:"):
                cols.append(raw[:, int(key[4:])])
            elif key == "sq1":
                cols.append(raw[:, 0] ** 2)
            elif key == "sq2":
                cols.append(raw[:, 1] ** 2)
            elif key == "prod":
                cols.append(raw[:, 0] * raw[:, 1])
            elif key == "sin1":
                cols.append(np.sin(raw[:, 0]))
            else:
                cols.append(np.sin(raw[:, 1]))
        return np.column_stack(cols) if cols else np.zeros((len(raw), 0))

    def fit(self, raw: np.ndarray) -> FeatureMap:
        f = self._raw(raw)
        if len(f) == 0 or f.shape[1] == 0:
            return self
        lo, hi = f.min(axis=0), f.max(axis=0)
        span = hi - lo
        span[span < 1e-12] = 1.0
        self.lo, self.span = lo, span
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Scaled features, in [-1, 1] on the data the map was fitted on."""
        return 2.0 * (self._raw(raw) - self.lo) / self.span - 1.0
