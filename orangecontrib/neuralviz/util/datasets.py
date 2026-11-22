"""Synthetic two-dimensional data sets, modelled on the ones in the TensorFlow Playground.

All coordinates lie in roughly [-6, 6]. ``noise`` is a percentage (0-50): the standard
deviation of the Gaussian jitter added to the coordinates, as a percentage of the data's
radius (6), so 0 gives clean data and 50 gives heavily overlapping classes.
"""

from __future__ import annotations

import numpy as np

RADIUS = 6.0

CLASSIFICATION = {
    "circle": "Circle",
    "xor": "XOR",
    "gauss": "Two Gaussians",
    "spiral": "Spiral",
}
REGRESSION = {
    "plane": "Plane",
    "multi_gauss": "Multiple Gaussians",
}
DATASETS = {**CLASSIFICATION, **REGRESSION}


def task_of(kind: str) -> str:
    """``"classification"`` or ``"regression"`` for a data set name."""
    if kind in CLASSIFICATION:
        return "classification"
    if kind in REGRESSION:
        return "regression"
    raise ValueError(f"Unknown data set '{kind}'")


def make_dataset(kind: str, n: int = 200, noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)``: ``X`` has shape ``(n, 2)``; ``y`` holds 0/1 labels or float values."""
    if kind not in DATASETS:
        raise ValueError(f"Unknown data set '{kind}'")
    n = max(2, int(n))
    rng = np.random.default_rng(seed)
    sigma = float(noise) / 100.0 * RADIUS

    if kind == "circle":
        n_in = n // 2
        r_in = rng.uniform(0.0, RADIUS * 0.5, n_in)
        r_out = rng.uniform(RADIUS * 0.7, RADIUS, n - n_in)
        r = np.concatenate([r_in, r_out])
        angle = rng.uniform(0.0, 2.0 * np.pi, n)
        X = np.column_stack([r * np.cos(angle), r * np.sin(angle)])
        y = np.concatenate([np.ones(n_in), np.zeros(n - n_in)])
    elif kind == "xor":
        pad = 0.3
        X = rng.uniform(-5.0, 5.0, size=(n, 2))
        X += np.sign(X) * pad
        y = (X[:, 0] * X[:, 1] >= 0).astype(float)
    elif kind == "gauss":
        n_pos = n // 2
        spread = 1.0 + 0.0
        pos = rng.normal([2.0, 2.0], spread, size=(n_pos, 2))
        neg = rng.normal([-2.0, -2.0], spread, size=(n - n_pos, 2))
        X = np.vstack([pos, neg])
        y = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)])
    elif kind == "spiral":
        n_pos = n // 2
        arms = []
        labels = []
        for label, count, offset in ((1.0, n_pos, 0.0), (0.0, n - n_pos, np.pi)):
            i = np.arange(count)
            r = i / max(count, 1) * RADIUS
            t = 1.75 * i / max(count, 1) * 2.0 * np.pi + offset
            arms.append(np.column_stack([r * np.sin(t), r * np.cos(t)]))
            labels.append(np.full(count, label))
        X = np.vstack(arms)
        y = np.concatenate(labels)
    elif kind == "plane":
        X = rng.uniform(-5.0, 5.0, size=(n, 2))
        y = (X[:, 0] + X[:, 1]) / 10.0
    else:  # multi_gauss
        X = rng.uniform(-RADIUS, RADIUS, size=(n, 2))
        centres = np.array([[-3.0, 3.0], [3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]])
        signs = np.array([1.0, -1.0, -1.0, 1.0])
        d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        y = (signs * np.exp(-d2 / (2.0 * 2.0**2))).sum(axis=1)

    if sigma > 0:
        X = X + rng.normal(0.0, sigma, size=X.shape)
    if task_of(kind) == "regression":
        y = np.asarray(y, dtype=float)
        if sigma > 0:
            y = y + rng.normal(0.0, sigma / RADIUS * 0.5, size=y.shape)
    else:
        y = y.astype(int)
    return X, y
