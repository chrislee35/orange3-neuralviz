"""A small feed-forward neural network in plain numpy.

Qt-free and Orange-free on purpose: it is fast enough for the tiny networks the
playground draws, easy to unit test, and has no dependency on TensorFlow or PyTorch.

Three output heads cover the three problems the widget supports:

* ``binary``     one sigmoid unit, binary cross-entropy
* ``softmax``    one unit per class, categorical cross-entropy
* ``regression`` one linear unit, squared error

In all three the gradient of the loss with respect to the output layer's
pre-activation is simply ``(prediction - target) / batch_size``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

HEADS = ("binary", "softmax", "regression")
REGULARIZATIONS = ("none", "l1", "l2")


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable


# name -> (function of the pre-activation, derivative written in terms of the *output*)
ACTIVATIONS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "relu": (_relu, lambda a: (a > 0).astype(a.dtype)),
    "tanh": (np.tanh, lambda a: 1.0 - a * a),
    "sigmoid": (_sigmoid, lambda a: a * (1.0 - a)),
    "linear": (lambda z: z, lambda a: np.ones_like(a)),
}


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class NeuralNetwork:
    """Fully connected network: ``n_inputs -> hidden[0] -> ... -> n_outputs``."""

    def __init__(
        self,
        n_inputs: int,
        hidden: list[int] | tuple[int, ...],
        n_outputs: int,
        activation: str = "tanh",
        head: str = "binary",
        seed: int = 0,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'")
        if head not in HEADS:
            raise ValueError(f"Unknown head '{head}'")
        if head != "softmax" and n_outputs != 1:
            raise ValueError(f"The '{head}' head needs exactly one output unit")
        if any(h < 1 for h in hidden):
            raise ValueError("Hidden layers need at least one neuron")
        self.sizes = [int(n_inputs), *[int(h) for h in hidden], int(n_outputs)]
        self.activation = activation
        self.head = head
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.reset(seed)

    # ----- parameters -------------------------------------------------------------------------

    def reset(self, seed: int = 0) -> None:
        """Re-initialise all weights (He for ReLU, Xavier otherwise) and biases."""
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        for i, (fan_in, fan_out) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            is_hidden_target = i < len(self.sizes) - 2
            if self.activation == "relu" and is_hidden_target:
                w = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
                b = np.full(fan_out, 0.1)  # keeps ReLU units alive at the start
            else:
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                w = rng.uniform(-limit, limit, size=(fan_in, fan_out))
                b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

    @property
    def n_layers(self) -> int:
        """Number of weight layers (hidden layers + 1)."""
        return len(self.weights)

    # ----- forward ----------------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        """Return the activations of every layer, input first, network output last."""
        act = ACTIVATIONS[self.activation][0]
        outs = [np.asarray(X, dtype=float)]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = outs[-1] @ w + b
            if i < self.n_layers - 1:
                outs.append(act(z))
            elif self.head == "binary":
                outs.append(_sigmoid(z))
            elif self.head == "softmax":
                outs.append(_softmax(z))
            else:
                outs.append(z)
        return outs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Network output: probabilities for the classification heads, values for regression."""
        return self.forward(X)[-1]

    # ----- loss -------------------------------------------------------------------------------

    def targets(self, y: np.ndarray) -> np.ndarray:
        """Turn a label/value vector into the matrix the loss compares the output with."""
        y = np.asarray(y)
        if self.head == "softmax":
            t = np.zeros((len(y), self.sizes[-1]))
            t[np.arange(len(y)), y.astype(int)] = 1.0
            return t
        return y.astype(float).reshape(-1, 1)

    def data_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean loss on (X, y), without the regularisation term."""
        if len(X) == 0:
            return float("nan")
        p = self.predict(X)
        t = self.targets(y)
        eps = 1e-12
        if self.head == "regression":
            return float(0.5 * np.mean((p - t) ** 2))
        if self.head == "binary":
            return float(-np.mean(t * np.log(p + eps) + (1.0 - t) * np.log(1.0 - p + eps)))
        return float(-np.mean(np.sum(t * np.log(p + eps), axis=1)))

    def regularization_loss(self, regularization: str, rate: float) -> float:
        if regularization == "l2":
            return float(0.5 * rate * sum(np.sum(w * w) for w in self.weights))
        if regularization == "l1":
            return float(rate * sum(np.sum(np.abs(w)) for w in self.weights))
        return 0.0

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correct class predictions (classification heads only)."""
        if len(X) == 0 or self.head == "regression":
            return float("nan")
        p = self.predict(X)
        pred = (p[:, 0] >= 0.5).astype(int) if self.head == "binary" else p.argmax(axis=1)
        return float(np.mean(pred == np.asarray(y).astype(int)))

    # ----- training ---------------------------------------------------------------------------

    def gradients(
        self, X: np.ndarray, y: np.ndarray, regularization: str = "none", rate: float = 0.0
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backpropagation. Regularisation acts on the weights only, never on the biases."""
        d_act = ACTIVATIONS[self.activation][1]
        outs = self.forward(X)
        delta = (outs[-1] - self.targets(y)) / len(X)
        g_w: list[np.ndarray] = [np.empty(0)] * self.n_layers
        g_b: list[np.ndarray] = [np.empty(0)] * self.n_layers
        for layer in reversed(range(self.n_layers)):
            g_w[layer] = outs[layer].T @ delta
            g_b[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * d_act(outs[layer])
        if regularization == "l2":
            g_w = [g + rate * w for g, w in zip(g_w, self.weights)]
        elif regularization == "l1":
            g_w = [g + rate * np.sign(w) for g, w in zip(g_w, self.weights)]
        return g_w, g_b

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        batch_size: int,
        rng: np.random.Generator,
        regularization: str = "none",
        rate: float = 0.0,
    ) -> None:
        """One pass over the data with mini-batch stochastic gradient descent."""
        if len(X) == 0:
            return
        order = rng.permutation(len(X))
        batch_size = max(1, int(batch_size))
        for start in range(0, len(X), batch_size):
            idx = order[start : start + batch_size]
            g_w, g_b = self.gradients(X[idx], y[idx], regularization, rate)
            for i in range(self.n_layers):
                self.weights[i] -= learning_rate * g_w[i]
                self.biases[i] -= learning_rate * g_b[i]
