import numpy as np
import pytest
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table

from orangecontrib.neuralviz.util.orange_io import data_with_subset, extract, generated_table, hidden_output, snapshot_model
from orangecontrib.neuralviz.util.session import Config, PlaygroundSession


def _session(table, **cfg):
    info = extract(table)
    names = [a.name for a in info.attributes]
    classes = list(info.class_var.values) if info.task == "classification" else None
    return info, PlaygroundSession(info.raw, info.y, info.task, names, classes, Config(**cfg), 0.5)


def test_generated_tables():
    t = generated_table("xor", 50, 5, 1)
    assert t.X.shape == (50, 2) and t.domain.class_var.is_discrete
    r = generated_table("plane", 50, 5, 1)
    assert r.domain.class_var.is_continuous


def test_extract_errors_and_cleaning():
    iris = Table("iris")
    info = extract(iris)
    assert info.task == "classification" and info.raw.shape == (150, 4) and info.dropped == 0

    with pytest.raises(ValueError, match="class"):
        extract(iris.transform(Domain(iris.domain.attributes)))
    with pytest.raises(ValueError, match="two numeric"):
        extract(iris.transform(Domain(iris.domain.attributes[:1], iris.domain.class_var)))

    t = generated_table("circle", 20, 0, 0)
    t.X[0, 0] = np.nan
    t.Y[1] = np.nan
    info = extract(t)
    assert info.dropped == 2 and len(info.raw) == 18 and 0 not in info.rows and 1 not in info.rows

    only_text = Domain([ContinuousVariable("a"), ContinuousVariable("b"), DiscreteVariable("d", values=("p", "q"))],
                       DiscreteVariable("c", values=("u", "v")))
    tt = Table.from_numpy(only_text, np.array([[1, 2, 0]] * 6, float), np.array([0, 1] * 3, float))
    assert [a.name for a in extract(tt).attributes] == ["a", "b"]  # discrete features are ignored

    tiny = generated_table("circle", 20, 0, 0)[:3]
    with pytest.raises(ValueError, match="rows"):
        extract(tiny)


def test_model_snapshot_predicts_like_the_network():
    table = generated_table("xor", 120, 0, 2)
    info, s = _session(table, hidden=[6], learning_rate=0.1, features=["col:0", "col:1", "prod"])
    s.step(60)
    model = snapshot_model(s, info)
    probs = model(table, model.Probs)
    assert probs.shape == (120, 2) and np.allclose(probs.sum(axis=1), 1)
    pred = model(table)
    assert np.mean(pred == table.Y) > 0.9
    # the snapshot is a copy: more training does not change it
    before = model(table, model.Probs).copy()
    s.step(20)
    assert np.array_equal(before, model(table, model.Probs))
    # missing inputs are imputed, not propagated
    nan_table = table.copy()
    nan_table.X[0, 0] = np.nan
    assert np.all(np.isfinite(model(nan_table, model.Probs)))


def test_model_regression_and_multiclass():
    plane = generated_table("plane", 100, 0, 1)
    info, s = _session(plane, hidden=[3], learning_rate=0.05)
    s.step(80)
    m = snapshot_model(s, info)
    assert m(plane).shape == (100,) and np.corrcoef(m(plane), plane.Y)[0, 1] > 0.95

    iris = Table("iris")
    info, s = _session(iris, hidden=[8], learning_rate=0.1, features=["col:0", "col:1", "col:2", "col:3"])
    s.step(150)
    m = snapshot_model(s, info)
    assert m(iris, m.Probs).shape == (150, 3) and np.mean(m(iris) == iris.Y) > 0.8


def test_output_tables():
    iris = Table("iris")
    info, s = _session(iris, hidden=[5, 3], features=["col:0", "col:1"])
    data = data_with_subset(s, info)
    assert len(data) == 150 and data.metas.shape == (150, 1)
    assert int(data.metas.sum()) == len(s.test_idx)
    hid = hidden_output(s, info)
    assert hid.X.shape == (150, 3) and [a.name for a in hid.domain.attributes] == ["h1", "h2", "h3"]
    assert hid.domain.class_var is info.class_var
    _, flat = _session(iris, hidden=[])
    assert hidden_output(flat, info) is None
