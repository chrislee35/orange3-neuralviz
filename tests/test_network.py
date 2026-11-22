import numpy as np
import pytest

from orangecontrib.neuralviz.util.datasets import DATASETS, make_dataset, task_of
from orangecontrib.neuralviz.util.features import FeatureMap
from orangecontrib.neuralviz.util.network import ACTIVATIONS, NeuralNetwork
from orangecontrib.neuralviz.util.session import Config, PlaygroundSession


def _numeric_grad(
    net: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    reg: str,
    rate: float,
    param: np.ndarray,
    idx: int | tuple[int, ...],
    eps: float = 1e-6,
) -> float:
    old = param[idx]
    param[idx] = old + eps
    up = net.data_loss(X, y) + net.regularization_loss(reg, rate)
    param[idx] = old - eps
    down = net.data_loss(X, y) + net.regularization_loss(reg, rate)
    param[idx] = old
    return (up - down) / (2 * eps)


@pytest.mark.parametrize("head,n_out", [("binary", 1), ("softmax", 3), ("regression", 1)])
@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "linear", "relu"])
@pytest.mark.parametrize("reg", ["none", "l1", "l2"])
def test_gradients_match_numeric(head: str, n_out: int, activation: str, reg: str) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 3))
    y = rng.integers(0, n_out if head == "softmax" else 2, 12) if head != "regression" else rng.normal(size=12)
    net = NeuralNetwork(3, [4, 3], n_out, activation, head, seed=2)
    rate = 0.05
    g_w, g_b = net.gradients(X, y, reg, rate)
    for layer in range(net.n_layers):
        for idx in [(0, 0), (1, net.weights[layer].shape[1] - 1)]:
            num = _numeric_grad(net, X, y, reg, rate, net.weights[layer], idx)
            assert g_w[layer][idx] == pytest.approx(num, abs=1e-5, rel=1e-3)
        num_b = _numeric_grad(net, X, y, reg, rate, net.biases[layer], 0)
        assert g_b[layer][0] == pytest.approx(num_b, abs=1e-5, rel=1e-3)


def test_forward_shapes_and_heads() -> None:
    X = np.zeros((5, 2))
    assert NeuralNetwork(2, [3], 1, head="binary").predict(X).shape == (5, 1)
    p = NeuralNetwork(2, [3], 4, head="softmax").predict(X)
    assert p.shape == (5, 4) and np.allclose(p.sum(axis=1), 1)
    assert [o.shape[1] for o in NeuralNetwork(2, [3, 5], 1).forward(X)] == [2, 3, 5, 1]
    assert NeuralNetwork(2, [], 1).n_layers == 1  # no hidden layers = logistic/linear model


def test_validation() -> None:
    with pytest.raises(ValueError):
        NeuralNetwork(2, [3], 1, activation="nope")
    with pytest.raises(ValueError):
        NeuralNetwork(2, [3], 2, head="binary")
    with pytest.raises(ValueError):
        NeuralNetwork(2, [0], 1)


def test_reset_is_reproducible() -> None:
    a, b = NeuralNetwork(2, [3], 1, seed=5), NeuralNetwork(2, [3], 1, seed=5)
    assert all(np.array_equal(x, y) for x, y in zip(a.weights, b.weights))
    a.reset(6)
    assert not np.array_equal(a.weights[0], b.weights[0])


def test_activation_derivatives() -> None:
    z = np.linspace(-2, 2, 9) + 0.013
    for name, (f, df) in ACTIVATIONS.items():
        num = (f(z + 1e-6) - f(z - 1e-6)) / 2e-6
        assert np.allclose(df(f(z)), num, atol=1e-5), name


def test_regularization_shrinks_weights() -> None:
    X, y = make_dataset("xor", 100, seed=1)
    X = X / 6
    norms = {}
    for reg in ("none", "l1", "l2"):
        net = NeuralNetwork(2, [4], 1, "tanh", "binary", seed=3)
        rng = np.random.default_rng(0)
        for _ in range(60):
            net.train_epoch(X, y, 0.1, 10, rng, reg, 0.3)
        norms[reg] = sum(np.abs(w).sum() for w in net.weights)
    assert norms["l1"] < norms["none"] and norms["l2"] < norms["none"]


@pytest.mark.parametrize("kind", ["xor", "circle", "gauss", "spiral"])
def test_learns_classification(kind: str) -> None:
    X, y = make_dataset(kind, 300, noise=0, seed=1)
    cfg = Config(
        hidden=[8, 8],
        activation="tanh",
        learning_rate=0.05,
        batch_size=10,
        features=["col:0", "col:1", "sq1", "sq2", "prod", "sin1", "sin2"],
        seed=1,
    )
    s = PlaygroundSession(X, y, "classification", ["x1", "x2"], ["0", "1"], cfg, train_fraction=0.5, split_seed=1)
    first = s.train_loss
    s.step(150 if kind != "spiral" else 400)
    assert s.train_loss < first
    assert s.train_accuracy > (0.9 if kind != "spiral" else 0.75)


def test_learns_regression_and_multiclass() -> None:
    X, y = make_dataset("plane", 200, seed=1)
    s = PlaygroundSession(X, y, "regression", ["x1", "x2"], None, Config(hidden=[3], learning_rate=0.03), 0.5)
    first = s.train_loss
    s.step(100)
    assert s.train_loss < first / 3

    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [5, 5], [-5, 5]])
    lab = rng.integers(0, 3, 150)
    Xm = centres[lab] + rng.normal(size=(150, 2))
    s = PlaygroundSession(Xm, lab, "classification", ["a", "b"], ["u", "v", "w"], Config(hidden=[6], learning_rate=0.05), 0.6)
    assert s.head == "softmax" and s.network.sizes[-1] == 3
    s.step(80)
    assert s.test_accuracy > 0.9


def test_session_split_history_and_live_settings() -> None:
    X, y = make_dataset("circle", 100, seed=0)
    s = PlaygroundSession(X, y, "classification", ["x1", "x2"], ["0", "1"], Config(), 0.7, split_seed=2)
    assert len(s.train_idx) == 70 and len(s.test_idx) == 30
    assert not set(s.train_idx) & set(s.test_idx)
    assert s.step(3) == 3 and s.epoch == 3 and len(s.history) == 4
    w_before = s.network.weights[0].copy()
    s.config.learning_rate = 0.0
    s.step(2)
    assert np.array_equal(w_before, s.network.weights[0])  # live setting took effect, nothing reset
    s.rebuild()
    assert s.epoch == 0 and len(s.history) == 1
    assert s.step(50, budget=0.0) == 1  # budget stops after the first epoch
    with pytest.raises(ValueError):
        PlaygroundSession(X, y, "classification", ["x1", "x2"], ["0"], Config())
    with pytest.raises(ValueError):
        PlaygroundSession(X, y, "classification", ["x1", "x2"], ["0", "1"], Config(features=[]))


def test_grid_activations() -> None:
    X, y = make_dataset("xor", 80, seed=0)
    s = PlaygroundSession(X, y, "classification", ["x1", "x2"], ["0", "1"], Config(hidden=[3, 2]), 0.5)
    acts = s.grid_activations(0, 1, 10)
    assert [a.shape for a in acts] == [(10, 10, 2), (10, 10, 3), (10, 10, 2), (10, 10, 1)]
    # image orientation: first row is the top (largest y), first column the left (smallest x)
    g = s.grid_raw(0, 1, 5)
    assert g[0, 1] > g[-1, 1] and g[0, 0] < g[4, 0]


def test_other_columns_held_at_mean() -> None:
    raw = np.random.default_rng(0).normal(size=(50, 4))
    y = (raw[:, 0] > 0).astype(int)
    s = PlaygroundSession(raw, y, "classification", list("abcd"), ["0", "1"], Config(features=["col:0", "col:2"]), 0.5)
    g = s.grid_raw(0, 2, 6)
    assert np.allclose(g[:, 1], raw[s.train_idx][:, 1].mean())


def test_feature_map() -> None:
    raw = np.array([[0.0, 0.0], [2.0, 4.0], [1.0, 2.0]])
    fm = FeatureMap(["a", "b"], ["col:0", "sq2", "prod", "sin1", "nope", "col:7"]).fit(raw)
    assert fm.selected == ["col:0", "sq2", "prod", "sin1"]
    assert fm.labels == ["a", "b²", "a·b", "sin(a)"]
    out = fm.transform(raw)
    assert out.min() == pytest.approx(-1) and out.max() == pytest.approx(1)
    one = FeatureMap(["a"], ["col:0", "sq1", "sq2", "prod"])
    assert one.selected == ["col:0", "sq1"]
    const = FeatureMap(["a", "b"], ["col:0"]).fit(np.ones((4, 2)))
    assert np.all(np.isfinite(const.transform(np.ones((4, 2)))))


@pytest.mark.parametrize("kind", list(DATASETS))
def test_datasets(kind: str) -> None:
    X, y = make_dataset(kind, 101, noise=10, seed=3)
    assert X.shape == (101, 2) and y.shape == (101,)
    assert np.all(np.abs(X) < 12)
    X2, y2 = make_dataset(kind, 101, noise=10, seed=3)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    if task_of(kind) == "classification":
        assert set(np.unique(y)) == {0, 1}
    else:
        assert y.dtype.kind == "f" and y.std() > 0.05
    with pytest.raises(ValueError):
        make_dataset("bogus")
