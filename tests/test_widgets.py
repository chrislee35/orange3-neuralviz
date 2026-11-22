import numpy as np
from Orange.data import Table
from orangewidget.tests.base import WidgetTest

from orangecontrib.neuralviz.widgets.owneural_playground import OWNeuralPlayground
from orangecontrib.neuralviz.widgets.owplayground_data import OWPlaygroundData


class TestPlaygroundData(WidgetTest):
    def test_generates(self):
        w = self.create_widget(OWPlaygroundData)
        out = self.get_output(w.Outputs.data, widget=w)
        assert len(out) == 200 and out.domain.class_var.is_discrete
        w.dataset_index = 4  # plane (regression)
        w.generate()
        assert self.get_output(w.Outputs.data, widget=w).domain.class_var.is_continuous
        w.reseed()
        assert w.seed == 1


class TestNeuralPlayground(WidgetTest):
    def setUp(self):
        self.w = self.create_widget(OWNeuralPlayground)

    def test_starts_with_builtin_data(self):
        w = self.w
        assert w.session is not None and w.session.epoch == 0
        assert not w.Error.active
        assert len(w.network_view.sizes) == 4  # features, 2 hidden, output
        assert self.get_output(w.Outputs.model, widget=w) is not None
        assert self.get_output(w.Outputs.hidden, widget=w) is not None

    def test_step_play_reset(self):
        w = self.w
        w._step_once()
        assert w.session.epoch == 1
        w._toggle_play()
        assert w.playing and w.play_button.text() == "Pause"
        w._tick()
        assert w.session.epoch == 1 + w.epochs_per_update
        w._toggle_play()
        assert not w.playing and w.play_button.text() == "Play"
        old = w.session.network.weights[0].copy()
        w._reset()
        assert w.session.epoch == 0 and not np.array_equal(old, w.session.network.weights[0])

    def test_trains_and_outputs_a_working_model(self):
        w = self.w
        w.dataset_index = 1  # xor
        w.feature_keys = ["col:0", "col:1", "prod"]
        w._on_data_changed()
        w.session.config.learning_rate = 0.1
        for _ in range(60):
            w._tick()
        w._toggle_play() if w.playing else None
        w._send_outputs()
        model = self.get_output(w.Outputs.model, widget=w)
        data = self.get_output(w.Outputs.data, widget=w)
        assert np.mean(model(data) == data.Y) > 0.9

    def test_live_settings_do_not_reset(self):
        w = self.w
        w._step_once()
        w.lr_index = 3
        w.reg_index = 2
        w.reg_rate_index = 4
        w.batch_size = 32
        w._on_live_changed()
        c = w.session.config
        assert (c.learning_rate, c.regularization, c.reg_rate, c.batch_size) == (0.003, "l2", 0.03, 32)
        assert w.session.epoch == 1

    def test_structure_changes_rebuild(self):
        w = self.w
        w._step_once()
        w.n_hidden = 3
        w._on_layer_count()
        assert w.session.network.sizes[1:-1] == [4, 2, 2] and w.session.epoch == 0
        w._on_neurons(0, 6)
        assert w.session.network.sizes[1:-1] == [6, 2, 2]
        w.n_hidden = 0
        w._on_layer_count()
        assert w.session.network.n_layers == 1
        assert self.get_output(w.Outputs.hidden, widget=w) is None
        w.activation_index = 0
        w._on_structure_changed()
        assert w.session.network.activation == "relu"

    def test_features(self):
        w = self.w
        assert w.feature_list.count() == 7
        for i in range(w.feature_list.count()):
            w.feature_list.item(i).setCheckState(0)  # Qt.Unchecked
        assert w.Error.no_features.is_shown() and w.session is None
        assert self.get_output(w.Outputs.model, widget=w) is None
        w.feature_list.item(0).setCheckState(2)
        assert not w.Error.no_features.is_shown() and w.session.network.sizes[0] == 1

    def test_input_table(self):
        w = self.w
        iris = Table("iris")
        self.send_signal(w.Inputs.data, iris, widget=w)
        assert w.session is not None and w.session.network.sizes[0] == 4
        assert w.session.network.sizes[-1] == 3 and w.session.head == "softmax"
        assert not w.generator_box.isVisibleTo(w)
        assert w.axis_x != w.axis_y
        w._step_once()
        out = self.get_output(w.Outputs.data, widget=w)
        assert len(out) == 150 and "Subset" in [m.name for m in out.domain.metas]
        self.send_signal(w.Inputs.data, None, widget=w)
        assert w.session.network.sizes[0] == 2 and w.generator_box.isVisibleTo(w)

    def test_bad_input_shows_error(self):
        w = self.w
        iris = Table("iris")
        no_class = iris.transform(type(iris.domain)(iris.domain.attributes))
        self.send_signal(w.Inputs.data, no_class, widget=w)
        assert w.Error.invalid_data.is_shown() and w.session is None
        assert self.get_output(w.Outputs.model, widget=w) is None
        self.send_signal(w.Inputs.data, iris, widget=w)
        assert not w.Error.invalid_data.is_shown() and w.session is not None

    def test_regression_and_missing_values(self):
        w = self.w
        w.dataset_index = 4
        w._on_data_changed()
        assert w.session.head == "regression"
        for _ in range(5):
            w._tick()
        t = Table("housing")[:60]
        t.X[0, 0] = np.nan
        self.send_signal(w.Inputs.data, t, widget=w)
        assert w.Information.dropped.is_shown() and w.session.head == "regression"
        w._step_once()

    def test_divergence_pauses(self):
        w = self.w
        w.lr_index = 10
        w.activation_index = 0
        w._on_structure_changed()
        w._on_live_changed()
        w.session.network.weights[0][:] = np.nan  # what a real blow-up leaves behind
        w._toggle_play()
        w._tick()
        assert not w.playing and w.Warning.diverged.is_shown()

    def test_hover_text(self):
        from AnyQt.QtCore import QPointF

        w = self.w
        w.network_view.resize(700, 500)
        w.network_view._layout()
        c = w.network_view._centres[1][0]
        assert "bias" in w.network_view.hover_text(c)
        a, b = w.network_view._edge_ends(0, 0, 0)
        mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
        assert "weight" in w.network_view.hover_text(mid)
        assert w.network_view.hover_text(QPointF(-50, -50)) is None

    def test_settings_roundtrip(self):
        w = self.w
        w.hidden_layers = [3]
        w.lr_index = 2
        saved = w.settingsHandler.pack_data(w)
        w2 = self.create_widget(OWNeuralPlayground, stored_settings=saved)
        assert w2.hidden_layers == [3] and w2.lr_index == 2 and w2.session.network.sizes[1:-1] == [3]
