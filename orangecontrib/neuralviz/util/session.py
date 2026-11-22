"""A training session: data split, features, network and the loss history.

Qt-free and Orange-free, so the widget stays thin and everything here can be unit tested.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from orangecontrib.neuralviz.util.features import FeatureMap, column_key
from orangecontrib.neuralviz.util.network import NeuralNetwork


@dataclass
class Config:
    """Everything the user can change. The first group needs a reset, the second applies live."""

    # Changing these rebuilds the network and starts training from scratch.
    hidden: list[int] = field(default_factory=lambda: [4, 2])
    activation: str = "tanh"
    features: list[str] = field(default_factory=lambda: [column_key(0), column_key(1)])
    seed: int = 0
    # Changing these takes effect immediately, mid-training.
    learning_rate: float = 0.03
    regularization: str = "none"
    reg_rate: float = 0.0
    batch_size: int = 10


class PlaygroundSession:
    """Owns one network being trained on one data set.

    ``raw`` is an ``(n, d)`` matrix of base columns. ``y`` holds class indices (0..k-1) for
    classification or floats for regression.
    """

    def __init__(
        self,
        raw: np.ndarray,
        y: np.ndarray,
        task: str,
        base_names: list[str],
        class_names: list[str] | None,
        config: Config,
        train_fraction: float = 0.5,
        split_seed: int = 0,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task '{task}'")
        self.raw = np.asarray(raw, dtype=float)
        self.y = np.asarray(y)
        self.task = task
        self.base_names = list(base_names)
        self.class_names = list(class_names) if class_names else []
        self.config = config
        if task == "classification":
            if len(self.class_names) < 2:
                raise ValueError("Classification needs at least two classes")
            self.head = "binary" if len(self.class_names) == 2 else "softmax"
            self.n_outputs = 1 if self.head == "binary" else len(self.class_names)
        else:
            self.head = "regression"
            self.n_outputs = 1

        rng = np.random.default_rng(split_seed)
        order = rng.permutation(len(self.raw))
        n_train = int(np.clip(round(train_fraction * len(self.raw)), 1, len(self.raw)))
        self.train_idx = np.sort(order[:n_train])
        self.test_idx = np.sort(order[n_train:])
        self.lo = self.raw.min(axis=0)
        self.hi = self.raw.max(axis=0)

        self.features: FeatureMap
        self.network: NeuralNetwork
        self.rng = np.random.default_rng(config.seed)
        self.epoch = 0
        self.history: list[tuple[int, float, float]] = []  # (epoch, train loss, test loss)
        self.rebuild()

    # ----- (re)building ----------------------------------------------------------------------

    def rebuild(self) -> None:
        """Fit the features, create a freshly initialised network and clear the history."""
        cfg = self.config
        self.features = FeatureMap(self.base_names, cfg.features).fit(self.raw[self.train_idx])
        if not self.features.selected:
            raise ValueError("Select at least one input feature")
        self.network = NeuralNetwork(
            len(self.features.selected), cfg.hidden, self.n_outputs, cfg.activation, self.head, cfg.seed
        )
        self.rng = np.random.default_rng(cfg.seed)
        self.epoch = 0
        self.history = []
        self.F_train = self.features.transform(self.raw[self.train_idx])
        self.F_test = self.features.transform(self.raw[self.test_idx])
        self.y_train = self.y[self.train_idx]
        self.y_test = self.y[self.test_idx]
        self._record()

    # ----- training --------------------------------------------------------------------------

    def _record(self) -> None:
        net = self.network
        self.history.append((self.epoch, net.data_loss(self.F_train, self.y_train), net.data_loss(self.F_test, self.y_test)))

    def step(self, epochs: int = 1, budget: float | None = None) -> int:
        """Train for up to ``epochs`` epochs, stopping early after ``budget`` seconds.

        Returns the number of epochs actually run (always at least one).
        """
        cfg = self.config
        start = time.perf_counter()
        done = 0
        for _ in range(max(1, int(epochs))):
            self.network.train_epoch(
                self.F_train, self.y_train, cfg.learning_rate, cfg.batch_size, self.rng, cfg.regularization, cfg.reg_rate
            )
            self.epoch += 1
            done += 1
            self._record()
            if budget is not None and time.perf_counter() - start > budget:
                break
        return done

    @property
    def train_loss(self) -> float:
        return self.history[-1][1]

    @property
    def test_loss(self) -> float:
        return self.history[-1][2]

    @property
    def train_accuracy(self) -> float:
        return self.network.accuracy(self.F_train, self.y_train)

    @property
    def test_accuracy(self) -> float:
        return self.network.accuracy(self.F_test, self.y_test)

    # ----- visualisation ---------------------------------------------------------------------

    def axis_range(self, axis: int, pad: float = 0.05) -> tuple[float, float]:
        lo, hi = float(self.lo[axis]), float(self.hi[axis])
        span = (hi - lo) or 1.0
        return lo - pad * span, hi + pad * span

    def grid_raw(self, axis_x: int, axis_y: int, resolution: int) -> np.ndarray:
        """Raw rows covering the (axis_x, axis_y) plane, other columns held at their mean.

        Row order is image order: ``y`` runs top to bottom (descending), ``x`` left to right.
        """
        x0, x1 = self.axis_range(axis_x)
        y0, y1 = self.axis_range(axis_y)
        xs = np.linspace(x0, x1, resolution)
        ys = np.linspace(y1, y0, resolution)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.tile(self.raw[self.train_idx].mean(axis=0), (resolution * resolution, 1))
        grid[:, axis_x] = gx.ravel()
        grid[:, axis_y] = gy.ravel()
        return grid

    def grid_activations(self, axis_x: int, axis_y: int, resolution: int = 40) -> list[np.ndarray]:
        """Activation of every layer over the plane: arrays of shape ``(res, res, units)``.

        The first entry is the (scaled) input features, the last the network output.
        """
        outs = self.network.forward(self.features.transform(self.grid_raw(axis_x, axis_y, resolution)))
        return [o.reshape(resolution, resolution, -1) for o in outs]
