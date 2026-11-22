"""Custom-painted Qt views: the network diagram, the output plane and the loss curve."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from AnyQt.QtCore import QPointF, QRectF, Qt
from AnyQt.QtGui import QColor, QFont, QFontMetrics, QImage, QMouseEvent, QPainter, QPen
from AnyQt.QtWidgets import QSizePolicy, QToolTip, QWidget

from orangecontrib.neuralviz.util.colors import BLUE, ORANGE
from orangecontrib.neuralviz.util.translate import _tr

_TEXT = QColor(70, 70, 70)
_FRAME = QColor(120, 120, 120)


def to_qimage(rgb: np.ndarray) -> QImage:
    """An ``(h, w, 3)`` uint8 array as a QImage that owns its pixels."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _qcolor(rgb: Sequence[float], alpha: int = 255) -> QColor:
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), alpha)


class NetworkView(QWidget):
    """Draws the layers as columns of small heat-map tiles joined by weighted edges.

    Blue edges have positive weights, orange ones negative; thicker means larger. Hover an edge
    for its weight and a tile for its bias.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.labels: list[str] = []
        self.sizes: list[int] = []
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.images: list[list[QImage]] = []
        self.headers: list[str] = []
        self.output_labels: list[str] = []
        self._centres: list[list[QPointF]] = []
        self._node = 30.0

    def set_state(
        self,
        labels: list[str],
        sizes: list[int],
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        images: list[list[QImage]],
        output_labels: list[str],
    ) -> None:
        self.labels = labels
        self.sizes = sizes
        self.weights = weights
        self.biases = biases
        self.images = images
        self.output_labels = output_labels
        self.headers = (
            [_tr.m[50, "Features"]] + [_tr.m[51, "Hidden {}"].format(i + 1) for i in range(len(sizes) - 2)] + [_tr.m[52, "Output"]]
        )
        self.update()

    # ----- geometry ---------------------------------------------------------------------------

    def _layout(self) -> None:
        w, h = self.width(), self.height()
        left, right, top, bottom = 96.0, (110.0 if len(self.output_labels) > 1 else 40.0), 30.0, 14.0
        n_cols = len(self.sizes)
        k_max = max(self.sizes) if self.sizes else 1
        avail = max(h - top - bottom, 10.0)
        self._node = float(np.clip(avail / k_max / 1.35, 16.0, 46.0))
        spacing = avail / k_max
        xs = np.linspace(left + self._node / 2, w - right - self._node / 2, n_cols) if n_cols > 1 else [w / 2]
        self._centres = []
        for col, k in enumerate(self.sizes):
            self._centres.append([QPointF(float(xs[col]), top + spacing * ((k_max - k) / 2 + j + 0.5)) for j in range(k)])

    def _edge_ends(self, layer: int, i: int, j: int) -> tuple[QPointF, QPointF]:
        half = self._node / 2
        a, b = self._centres[layer][i], self._centres[layer + 1][j]
        return QPointF(a.x() + half, a.y()), QPointF(b.x() - half, b.y())

    # ----- painting ---------------------------------------------------------------------------

    def paintEvent(self, _event: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self.sizes:
            return
        self._layout()
        for layer, w in enumerate(self.weights):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    value = float(w[i, j])
                    color = _qcolor(BLUE if value >= 0 else ORANGE, 170)
                    p.setPen(QPen(color, 0.5 + min(abs(value), 3.0) * 1.6))
                    a, b = self._edge_ends(layer, i, j)
                    p.drawLine(a, b)
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, min(9.0, self._node / 4.5)))
        p.setFont(font)
        half = self._node / 2
        for col, centres in enumerate(self._centres):
            p.setPen(_TEXT)
            if centres:
                p.drawText(QRectF(centres[0].x() - 60, 4, 120, 18), Qt.AlignmentFlag.AlignCenter, self.headers[col])
            for j, c in enumerate(centres):
                rect = QRectF(c.x() - half, c.y() - half, self._node, self._node)
                if col < len(self.images) and j < len(self.images[col]):
                    p.drawImage(rect, self.images[col][j])
                p.setPen(QPen(_FRAME, 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(rect)
                p.setPen(_TEXT)
                if col == 0 and j < len(self.labels):
                    p.drawText(
                        QRectF(0, c.y() - 10, c.x() - half - 4, 20),
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        self.labels[j],
                    )
                elif col == len(self._centres) - 1 and j < len(self.output_labels) and len(centres) > 1:
                    label = QFontMetrics(font).elidedText(self.output_labels[j], Qt.TextElideMode.ElideRight, 100)
                    p.drawText(
                        QRectF(c.x() + half + 4, c.y() - 10, 104, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label
                    )
        p.end()

    # ----- hover ------------------------------------------------------------------------------

    def hover_text(self, pos: QPointF) -> str | None:
        """Description of the tile or edge under ``pos``, or None."""
        if not self._centres:
            return None
        half = self._node / 2
        for col, centres in enumerate(self._centres):
            if col == 0:
                continue
            for j, c in enumerate(centres):
                if abs(pos.x() - c.x()) <= half and abs(pos.y() - c.y()) <= half:
                    return _tr.m[53, "{}, neuron {}: bias {}"].format(self.headers[col], j + 1, f"{self.biases[col - 1][j]:+.3f}")
        best, best_d = None, 5.0
        for layer, w in enumerate(self.weights):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    a, b = self._edge_ends(layer, i, j)
                    d = _segment_distance(pos, a, b)
                    if d < best_d:
                        best, best_d = (layer, i, j), d
        if best is None:
            return None
        layer, i, j = best
        src = self.labels[i] if layer == 0 and i < len(self.labels) else f"{self.headers[layer]} #{i + 1}"
        return _tr.m[54, "weight {} → {} #{}: {}"].format(src, self.headers[layer + 1], j + 1, f"{self.weights[layer][i, j]:+.3f}")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # Qt5 and Qt6 name these differently.
        local = event.localPos() if hasattr(event, "localPos") else event.position()
        global_pos = event.globalPos() if hasattr(event, "globalPos") else event.globalPosition().toPoint()
        text = self.hover_text(local)
        if text:
            QToolTip.showText(global_pos, text, self)
        else:
            QToolTip.hideText()


def _segment_distance(p: QPointF, a: QPointF, b: QPointF) -> float:
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p.x() - ax) * dx + (p.y() - ay) * dy) / length2))
    return math.hypot(p.x() - (ax + t * dx), p.y() - (ay + t * dy))


class OutputView(QWidget):
    """The network's output over the plane, with the training and test points on top."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image: QImage | None = None
        self.extent = (0.0, 1.0, 0.0, 1.0)
        self.train_xy = np.zeros((0, 2))
        self.train_rgb = np.zeros((0, 3), dtype=np.uint8)
        self.test_xy = np.zeros((0, 2))
        self.test_rgb = np.zeros((0, 3), dtype=np.uint8)
        self.show_test = True
        self.axis_names = ("x", "y")

    def set_state(
        self,
        image: QImage,
        extent: tuple[float, float, float, float],
        train_xy: np.ndarray,
        train_rgb: np.ndarray,
        test_xy: np.ndarray,
        test_rgb: np.ndarray,
        show_test: bool,
        axis_names: tuple[str, str],
    ) -> None:
        self.image, self.extent = image, extent
        self.train_xy, self.train_rgb = train_xy, train_rgb
        self.test_xy, self.test_rgb = test_xy, test_rgb
        self.show_test, self.axis_names = show_test, axis_names
        self.update()

    def _square(self) -> QRectF:
        margin_left, margin_bottom, margin = 22.0, 20.0, 6.0
        side = max(10.0, min(self.width() - margin_left - margin, self.height() - margin_bottom - margin))
        return QRectF(margin_left, margin, side, side)

    def paintEvent(self, _event: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        box = self._square()
        if self.image is not None:
            p.drawImage(box, self.image)
        p.setPen(QPen(_FRAME, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(box)
        x0, x1, y0, y1 = self.extent

        def to_px(xy: np.ndarray) -> list[QPointF]:
            return [
                QPointF(box.left() + (x - x0) / (x1 - x0) * box.width(), box.bottom() - (y - y0) / (y1 - y0) * box.height()) for x, y in xy
            ]

        radius = 3.2
        p.setClipRect(box)
        groups = [(self.train_xy, self.train_rgb, QPen(QColor(255, 255, 255), 1.0))]
        if self.show_test:
            groups.append((self.test_xy, self.test_rgb, QPen(QColor(20, 20, 20), 1.6)))
        for xy, rgb, pen in groups:
            p.setPen(pen)
            for pt, c in zip(to_px(xy), rgb):
                p.setBrush(_qcolor(c))
                p.drawEllipse(pt, radius, radius)
        p.setClipping(False)
        p.setPen(_TEXT)
        font = QFont(self.font())
        font.setPointSizeF(8.0)
        p.setFont(font)
        p.drawText(
            QRectF(box.left(), box.bottom() + 2, box.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            f"{self.axis_names[0]}  ({x0:.1f} … {x1:.1f})",
        )
        p.save()
        p.translate(12, box.bottom() - box.height() / 2)
        p.rotate(-90)
        p.drawText(
            QRectF(-box.height() / 2, -10, box.height(), 16), Qt.AlignmentFlag.AlignCenter, f"{self.axis_names[1]}  ({y0:.1f} … {y1:.1f})"
        )
        p.restore()
        p.end()


class LossView(QWidget):
    """Training (dark) and test (orange) loss against epoch."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(260, 110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.history: list[tuple[int, float, float]] = []

    def set_history(self, history: list[tuple[int, float, float]]) -> None:
        self.history = history
        self.update()

    def paintEvent(self, _event: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = QRectF(40, 6, max(self.width() - 50, 10), max(self.height() - 24, 10))
        p.setPen(QPen(_FRAME, 1))
        p.drawRect(box)
        h = self.history
        if len(h) >= 2:
            step = max(1, len(h) // 400)  # thin out very long histories
            pts = h[::step] if h[::step][-1] is h[-1] else h[::step] + [h[-1]]
            epochs = np.array([e for e, _, _ in pts], dtype=float)
            series = [np.array([t for _, t, _ in pts]), np.array([v for _, _, v in pts])]
            finite = np.concatenate([s[np.isfinite(s)] for s in series])
            top = float(finite.max()) if len(finite) else 1.0
            top = top if top > 0 else 1.0
            span = max(epochs[-1] - epochs[0], 1.0)
            for values, color in zip(series, (QColor(50, 50, 50), QColor(*[int(c) for c in ORANGE]))):
                p.setPen(QPen(color, 1.6))
                last: QPointF | None = None
                for e, v in zip(epochs, values):
                    if not np.isfinite(v):
                        last = None
                        continue
                    pt = QPointF(box.left() + (e - epochs[0]) / span * box.width(), box.bottom() - min(v / top, 1.0) * box.height())
                    if last is not None:
                        p.drawLine(last, pt)
                    last = pt
            p.setPen(_TEXT)
            font = QFont(self.font())
            font.setPointSizeF(8.0)
            p.setFont(font)
            p.drawText(QRectF(0, box.top() - 2, 38, 14), Qt.AlignmentFlag.AlignRight, f"{top:.2g}")
            p.drawText(QRectF(0, box.bottom() - 12, 38, 14), Qt.AlignmentFlag.AlignRight, "0")
            p.drawText(
                QRectF(box.left(), box.bottom() + 2, box.width(), 14),
                Qt.AlignmentFlag.AlignCenter,
                _tr.m[55, "epoch {} … {}"].format(int(epochs[0]), int(epochs[-1])),
            )
        p.end()
