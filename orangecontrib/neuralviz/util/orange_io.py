"""Glue between the Qt-free engine and Orange's data tables and models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from Orange.classification import Model as ClassificationModel
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table, Variable
from Orange.regression import Model as RegressionModel

from orangecontrib.neuralviz.util.datasets import make_dataset, task_of
from orangecontrib.neuralviz.util.features import FeatureMap
from orangecontrib.neuralviz.util.network import NeuralNetwork
from orangecontrib.neuralviz.util.session import PlaygroundSession

SUBSET_NAME = "Subset"
MIN_ROWS = 4


@dataclass
class Extracted:
    """The numeric part of an Orange table, ready for a :class:`PlaygroundSession`."""

    raw: np.ndarray
    y: np.ndarray
    task: str
    attributes: list[Variable]
    class_var: Variable
    rows: np.ndarray  # indices into the original table of the rows that were kept
    dropped: int  # rows left out because of missing values


def extract(data: Table) -> Extracted:
    """Pull base columns, target and task out of ``data``. Raises ``ValueError`` with a user-facing message."""
    class_var = data.domain.class_var
    if class_var is None:
        raise ValueError("The data needs a class (target) variable.")
    if len(data.domain.class_vars) > 1:
        raise ValueError("Data with several class variables is not supported.")
    attributes = [a for a in data.domain.attributes if a.is_continuous]
    if len(attributes) < 2:
        raise ValueError("The data needs at least two numeric features.")
    if class_var.is_discrete and len(class_var.values) < 2:
        raise ValueError("The class variable needs at least two values.")
    if not (class_var.is_discrete or class_var.is_continuous):
        raise ValueError("The class variable must be categorical or numeric.")
    sub = data.transform(Domain(attributes, class_var))
    ok = ~(np.isnan(sub.X).any(axis=1) | np.isnan(sub.Y))
    if ok.sum() < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} rows without missing values are needed.")
    rows = np.flatnonzero(ok)
    y = sub.Y[ok]
    task = "classification" if class_var.is_discrete else "regression"
    return Extracted(
        np.asarray(sub.X[ok], dtype=float),
        y.astype(int) if task == "classification" else y.astype(float),
        task,
        attributes,
        class_var,
        rows,
        int((~ok).sum()),
    )


def generated_table(kind: str, n: int, noise: float, seed: int) -> Table:
    """A synthetic playground data set as an Orange table with ``x1``, ``x2`` and a class."""
    X, y = make_dataset(kind, n, noise, seed)
    class_var: Variable = DiscreteVariable("class", values=("0", "1")) if task_of(kind) == "classification" else ContinuousVariable("value")
    domain = Domain([ContinuousVariable("x1"), ContinuousVariable("x2")], class_var)
    table = Table.from_numpy(domain, X, y.astype(float))
    table.name = {"classification": "Playground classification", "regression": "Playground regression"}[task_of(kind)]
    return table


def _subset_var() -> DiscreteVariable:
    return DiscreteVariable(SUBSET_NAME, values=("train", "test"))


def data_with_subset(session: PlaygroundSession, info: Extracted) -> Table:
    """The rows the session trains on, tagged as train or test."""
    subset = np.zeros(len(session.raw))
    subset[session.test_idx] = 1.0
    domain = Domain(info.attributes, info.class_var, metas=[_subset_var()])
    return Table.from_numpy(domain, session.raw, info.y.astype(float), subset.reshape(-1, 1))


def hidden_output(session: PlaygroundSession, info: Extracted) -> Table | None:
    """Activations of the last hidden layer for every row: what the network 'sees' just before the output."""
    if session.network.n_layers < 2:
        return None
    acts = session.network.forward(session.features.transform(session.raw))[-2]
    subset = np.zeros(len(session.raw))
    subset[session.test_idx] = 1.0
    attrs = [ContinuousVariable(f"h{i + 1}") for i in range(acts.shape[1])]
    domain = Domain(attrs, info.class_var, metas=[_subset_var()])
    return Table.from_numpy(domain, acts, info.y.astype(float), subset.reshape(-1, 1))


class _SnapshotMixin:
    """Prediction logic shared by the classification and regression model classes."""

    network: NeuralNetwork
    features: FeatureMap
    means: np.ndarray

    def _outputs(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        nan = np.isnan(X)
        if nan.any():  # impute with the training means, as most Orange models do
            X[nan] = np.broadcast_to(self.means, X.shape)[nan]
        return self.network.predict(self.features.transform(X))


class NeuralClassifier(_SnapshotMixin, ClassificationModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self._outputs(X)
        return np.column_stack([1.0 - p[:, 0], p[:, 0]]) if self.network.head == "binary" else p


class NeuralRegressor(_SnapshotMixin, RegressionModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._outputs(X)[:, 0]


def snapshot_model(session: PlaygroundSession, info: Extracted) -> Any:
    """An Orange model holding a copy of the network as it is right now (training can carry on)."""
    domain = Domain(info.attributes, info.class_var)
    cls = NeuralClassifier if info.task == "classification" else NeuralRegressor
    model = cls(domain)
    model.network = copy.deepcopy(session.network)
    model.features = copy.deepcopy(session.features)
    model.means = session.raw[session.train_idx].mean(axis=0)
    model.name = "Neural Network"
    return model
