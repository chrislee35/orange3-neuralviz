"""Colour maps for the playground. Numpy only; the views turn the arrays into Qt images."""

from __future__ import annotations

import numpy as np

ORANGE = np.array([245, 136, 58], dtype=float)  # negative values / class 0
BLUE = np.array([47, 127, 193], dtype=float)  # positive values / class 1
WHITE = np.array([255, 255, 255], dtype=float)

CLASS_COLORS = np.array(
    [
        ORANGE,
        BLUE,
        [76, 175, 80],
        [156, 39, 176],
        [229, 57, 53],
        [121, 85, 72],
        [233, 30, 99],
        [96, 125, 139],
        [158, 158, 36],
        [0, 172, 193],
    ],
    dtype=float,
)


def _u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def diverging(values: np.ndarray) -> np.ndarray:
    """White at 0, blue towards +1, orange towards -1 (values are clipped to [-1, 1])."""
    v = np.clip(np.asarray(values, dtype=float), -1.0, 1.0)[..., None]
    target = np.where(v >= 0, BLUE, ORANGE)
    return _u8(WHITE + (target - WHITE) * np.abs(v))


def white_to(values: np.ndarray, class_index: int) -> np.ndarray:
    """White at 0 to the colour of one class at 1."""
    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)[..., None]
    return _u8(WHITE + (CLASS_COLORS[class_index % len(CLASS_COLORS)] - WHITE) * v)


def class_blend(probs: np.ndarray) -> np.ndarray:
    """Mix the class colours by their probabilities, lightened so points stand out on top."""
    p = np.asarray(probs, dtype=float)
    colors = CLASS_COLORS[np.arange(p.shape[-1]) % len(CLASS_COLORS)]
    rgb = p @ colors
    return _u8(rgb + (WHITE - rgb) * 0.35)


def class_colors(labels: np.ndarray) -> np.ndarray:
    """Palette colour of each class label."""
    return _u8(CLASS_COLORS[np.asarray(labels, dtype=int) % len(CLASS_COLORS)])


def output_image(output: np.ndarray, head: str, value_scale: float = 1.0) -> np.ndarray:
    """Colourise a ``(h, w, k)`` network output: the decision map shown behind the data."""
    if head == "binary":
        return diverging(2.0 * output[..., 0] - 1.0)
    if head == "softmax":
        return class_blend(output)
    return diverging(output[..., 0] / value_scale)


def point_colors(y: np.ndarray, head: str, value_scale: float = 1.0) -> np.ndarray:
    """Colour of each data point: its class, or its value for regression."""
    if head == "regression":
        return diverging(np.asarray(y, dtype=float) / value_scale)
    return class_colors(y)
