from __future__ import annotations

from functools import partial

import numpy as np
from AnyQt.QtCore import Qt, QTimer
from AnyQt.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QSpinBox, QWidget
from Orange.base import Model
from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.widget import Input, Msg, Output

from orangecontrib.neuralviz.util.colors import diverging, output_image, point_colors, white_to
from orangecontrib.neuralviz.util.datasets import DATASETS
from orangecontrib.neuralviz.util.features import DERIVED_KEYS, FeatureMap, column_key
from orangecontrib.neuralviz.util.orange_io import (
    Extracted,
    data_with_subset,
    extract,
    generated_table,
    hidden_output,
    snapshot_model,
)
from orangecontrib.neuralviz.util.session import Config, PlaygroundSession
from orangecontrib.neuralviz.util.typed_settings import setting
from orangecontrib.neuralviz.util.views import LossView, NetworkView, OutputView, to_qimage

DATASET_KEYS = list(DATASETS)
ACTIVATION_KEYS = ["relu", "tanh", "sigmoid", "linear"]
ACTIVATION_NAMES = ["ReLU", "Tanh", "Sigmoid", "Linear"]
LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
REGULARIZATION_KEYS = ["none", "l1", "l2"]
REGULARIZATION_RATES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
MAX_HIDDEN_LAYERS = 6
MAX_NEURONS = 8
GRID = 40
TICK_MS = 30
TICK_BUDGET = 0.025  # seconds of training per tick, so the window stays responsive


class OWNeuralPlayground(widget.OWWidget):
    name = "Neural Playground"
    description = "Build a small neural network, train it step by step and watch what it learns."
    icon = "icons/nv-playground.svg"
    priority = 10
    keywords = ["neural network", "playground", "perceptron", "mlp", "deep learning", "tensorflow", "train"]
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        model = Output("Model", Model)
        data = Output("Data", Table)
        hidden = Output("Hidden Layer Output", Table)

    class Error(widget.OWWidget.Error):
        invalid_data = Msg("{}")
        no_features = Msg("Select at least one input feature.")

    class Warning(widget.OWWidget.Warning):
        diverged = Msg("Training diverged (the loss is no longer finite). Lower the learning rate and press Reset.")

    class Information(widget.OWWidget.Information):
        dropped = Msg("{} rows with missing values were ignored.")

    # data
    dataset_index: int = setting(0)
    n_samples: int = setting(200)
    noise: int = setting(0)
    data_seed: int = setting(0)
    train_percent: int = setting(50)
    # network
    feature_keys: list[str] = setting([column_key(0), column_key(1)])
    hidden_layers: list[int] = setting([4, 2])
    activation_index: int = setting(1)
    # training
    lr_index: int = setting(5)
    reg_index: int = setting(0)
    reg_rate_index: int = setting(0)
    batch_size: int = setting(10)
    epochs_per_update: int = setting(1)
    # display
    axis_x: int = setting(0)
    axis_y: int = setting(1)
    show_test: bool = setting(True)

    def __init__(self) -> None:
        super().__init__()
        self.input_data: Table | None = None
        self.extracted: Extracted | None = None
        self.session: PlaygroundSession | None = None
        self.init_seed = 0
        self.value_scale = 1.0
        self.playing = False
        self._signature: tuple[str, ...] = ()
        self.hidden_layers = [max(1, min(MAX_NEURONS, int(n))) for n in self.hidden_layers][:MAX_HIDDEN_LAYERS]
        self.feature_keys = list(self.feature_keys)
        self.n_hidden = len(self.hidden_layers)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)

        self._build_controls()
        self._build_main_area()
        self.resize(1180, 800)
        self._load()

    # ----- layout -----------------------------------------------------------------------------

    def _build_controls(self) -> None:
        row_box = gui.hBox(self.controlArea)
        gui.button(row_box, self, "Reset", callback=self._reset)
        self.play_button = gui.button(row_box, self, "Play", callback=self._toggle_play)
        gui.button(row_box, self, "Step", callback=self._step_once)

        box = gui.vBox(self.controlArea, "Data")
        self.data_info = gui.widgetLabel(box, "")
        self.generator_box = gui.vBox(box)
        gui.comboBox(
            self.generator_box, self, "dataset_index", items=list(DATASETS.values()), orientation=Qt.Orientation.Horizontal,
            callback=self._on_data_changed,
        )
        gui.spin(self.generator_box, self, "n_samples", 50, 1000, step=50, label="Samples:", callback=self._on_data_changed)
        gui.hSlider(
            self.generator_box, self, "noise", label="Noise:", minValue=0, maxValue=50, step=1, intOnly=True, labelFormat=" %d%%",
            callback=self._on_data_changed,
        )
        gui.button(self.generator_box, self, "New random data", callback=self._regenerate)
        gui.hSlider(
            box, self, "train_percent", label="Train:", minValue=10, maxValue=90, step=5, intOnly=True, labelFormat=" %d%%",
            callback=self._on_structure_changed,
        )

        box = gui.vBox(self.controlArea, "Input features")
        self.feature_list = QListWidget()
        self.feature_list.setMaximumHeight(120)
        self.feature_list.itemChanged.connect(self._on_feature_toggled)
        box.layout().addWidget(self.feature_list)

        box = gui.vBox(self.controlArea, "Network")
        gui.spin(box, self, "n_hidden", 0, MAX_HIDDEN_LAYERS, label="Hidden layers:", callback=self._on_layer_count)
        row = QWidget()
        self.neuron_layout = QHBoxLayout(row)
        self.neuron_layout.setContentsMargins(0, 0, 0, 0)
        box.layout().addWidget(row)
        self._populate_neurons()
        gui.comboBox(
            box, self, "activation_index", label="Activation:", items=ACTIVATION_NAMES, orientation=Qt.Orientation.Horizontal,
            callback=self._on_structure_changed,
        )

        box = gui.vBox(self.controlArea, "Training")
        gui.comboBox(
            box, self, "lr_index", label="Learning rate:", items=[f"{v:g}" for v in LEARNING_RATES],
            orientation=Qt.Orientation.Horizontal, callback=self._on_live_changed,
        )
        gui.comboBox(
            box, self, "reg_index", label="Regularization:", items=["None", "L1", "L2"], orientation=Qt.Orientation.Horizontal,
            callback=self._on_live_changed,
        )
        gui.comboBox(
            box, self, "reg_rate_index", label="Reg. rate:", items=[f"{v:g}" for v in REGULARIZATION_RATES],
            orientation=Qt.Orientation.Horizontal, callback=self._on_live_changed,
        )
        gui.spin(box, self, "batch_size", 1, 512, label="Batch size:", callback=self._on_live_changed)
        gui.spin(box, self, "epochs_per_update", 1, 50, label="Epochs per update:")

        box = gui.vBox(self.controlArea, "Display")
        self.axes_box = gui.vBox(box)
        self.axis_x_combo = gui.comboBox(
            self.axes_box, self, "axis_x", label="Horizontal axis:", items=[], orientation=Qt.Orientation.Horizontal, callback=self._on_axes_changed
        )
        self.axis_y_combo = gui.comboBox(
            self.axes_box, self, "axis_y", label="Vertical axis:", items=[], orientation=Qt.Orientation.Horizontal, callback=self._on_axes_changed
        )
        gui.checkBox(box, self, "show_test", "Show test data", callback=self._refresh)


    def _build_main_area(self) -> None:
        holder = gui.hBox(self.mainArea)
        self.network_view = NetworkView()
        holder.layout().addWidget(self.network_view, 3)
        right = gui.vBox(holder)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.output_view = OutputView()
        self.loss_view = LossView()
        right.layout().addWidget(self.stats)
        right.layout().addWidget(self.output_view, 3)
        right.layout().addWidget(self.loss_view, 1)
        holder.layout().addWidget(right, 2)

    # ----- control population ----------------------------------------------------------------

    def _populate_features(self) -> None:
        info = self.extracted
        self.feature_list.blockSignals(True)
        self.feature_list.clear()
        if info is not None:
            names = [a.name for a in info.attributes]
            fm = FeatureMap(names, [])
            keys = [column_key(i) for i in range(len(names))] + [k for k in DERIVED_KEYS if fm.is_available(k)]
            fm = FeatureMap(names, keys)
            for key in keys:
                item = QListWidgetItem(fm.label(key))
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if key in self.feature_keys else Qt.CheckState.Unchecked)
                self.feature_list.addItem(item)
        self.feature_list.blockSignals(False)

    def _populate_neurons(self) -> None:
        while self.neuron_layout.count():
            item = self.neuron_layout.takeAt(0)
            if item is not None and item.widget() is not None:
                item.widget().deleteLater()
        for i, n in enumerate(self.hidden_layers):
            spin = QSpinBox()
            spin.setRange(1, MAX_NEURONS)
            spin.setValue(n)
            spin.setToolTip(f"Neurons in hidden layer {i + 1}")
            spin.valueChanged.connect(partial(self._on_neurons, i))
            self.neuron_layout.addWidget(spin)
        self.neuron_layout.addStretch(1)

    def _populate_axes(self) -> None:
        names = [a.name for a in self.extracted.attributes] if self.extracted else []
        for combo in (self.axis_x_combo, self.axis_y_combo):
            combo.clear()
            combo.addItems(names)
        n = len(names)
        self.axes_box.setVisible(n > 2)  # with two columns the plane is simply x1 against x2
        if n == 0:
            return
        if not (0 <= self.axis_x < n):
            self.axis_x = 0
        if not (0 <= self.axis_y < n) or self.axis_y == self.axis_x:
            self.axis_y = 1 if self.axis_x != 1 else 0
        self.axis_x_combo.setCurrentIndex(self.axis_x)
        self.axis_y_combo.setCurrentIndex(self.axis_y)

    # ----- data -------------------------------------------------------------------------------

    @Inputs.data
    def set_data(self, data: Table | None) -> None:
        self.input_data = data

    def handleNewSignals(self) -> None:  # noqa: N802
        self._pause()
        self._load()

    def _load(self) -> None:
        self.Error.clear()
        self.Information.clear()
        self.Warning.clear()
        self.session = None
        self.extracted = None
        self.generator_box.setVisible(self.input_data is None)
        try:
            if self.input_data is not None:
                self.extracted = extract(self.input_data)
                if self.extracted.dropped:
                    self.Information.dropped(self.extracted.dropped)
            else:
                self.extracted = extract(generated_table(DATASET_KEYS[self.dataset_index], self.n_samples, self.noise, self.data_seed))
        except ValueError as e:
            self.Error.invalid_data(str(e))
            self.data_info.setText("")
            self.data_info.setVisible(False)
            self._populate_features()
            self._populate_axes()
            self._clear_views()
            self._send_outputs()
            return

        names = tuple(a.name for a in self.extracted.attributes)
        if names != self._signature:  # a different set of columns: pick sensible default inputs
            self._signature = names
            if self.input_data is not None:
                self.feature_keys = [column_key(i) for i in range(len(names) if len(names) <= 8 else 2)]
            else:
                self.feature_keys = [column_key(0), column_key(1)]
        if self.input_data is not None:
            kind = "classes" if self.extracted.task == "classification" else "regression"
            n_classes = len(self.extracted.class_var.values) if self.extracted.task == "classification" else 0  # type: ignore[union-attr]
            detail = f"{n_classes} {kind}" if n_classes else "numeric target"
            self.data_info.setText(f"{len(self.extracted.raw)} rows, {len(names)} numeric features, {detail}")
        else:
            self.data_info.setText("")
        self.data_info.setVisible(self.input_data is not None)
        self._populate_features()
        self._populate_axes()
        self._build_session()

    def _build_session(self) -> None:
        info = self.extracted
        self.Error.no_features.clear()
        self.Warning.diverged.clear()
        if info is None:
            return
        names = [a.name for a in info.attributes]
        if not FeatureMap(names, self.feature_keys).selected:
            self.session = None
            self.Error.no_features()
            self._clear_views()
            self._send_outputs()
            return
        class_names = list(info.class_var.values) if info.task == "classification" else None  # type: ignore[union-attr]
        config = Config(
            hidden=list(self.hidden_layers),
            activation=ACTIVATION_KEYS[self.activation_index],
            features=list(self.feature_keys),
            seed=self.init_seed,
            learning_rate=LEARNING_RATES[self.lr_index],
            regularization=REGULARIZATION_KEYS[self.reg_index],
            reg_rate=REGULARIZATION_RATES[self.reg_rate_index],
            batch_size=self.batch_size,
        )
        self.session = PlaygroundSession(
            info.raw, info.y, info.task, names, class_names, config, self.train_percent / 100.0, split_seed=self.data_seed
        )
        self.value_scale = max(float(np.abs(info.y).max()), 1e-9) if info.task == "regression" else 1.0
        self._refresh()
        self._send_outputs()

    # ----- control callbacks -----------------------------------------------------------------

    def _on_data_changed(self) -> None:
        self._pause()
        self._load()

    def _regenerate(self) -> None:
        self.data_seed = (self.data_seed + 1) % 100000
        self._on_data_changed()

    def _on_structure_changed(self) -> None:
        self._pause()
        self._build_session()

    def _on_feature_toggled(self, item: QListWidgetItem) -> None:
        keys = []
        for i in range(self.feature_list.count()):
            it = self.feature_list.item(i)
            if it is not None and it.checkState() == Qt.CheckState.Checked:
                keys.append(it.data(Qt.ItemDataRole.UserRole))
        self.feature_keys = keys
        self._on_structure_changed()

    def _on_layer_count(self) -> None:
        n = self.n_hidden
        layers = list(self.hidden_layers[:n])
        while len(layers) < n:
            layers.append(layers[-1] if layers else 4)
        self.hidden_layers = layers
        self._populate_neurons()
        self._on_structure_changed()

    def _on_neurons(self, layer: int, value: int) -> None:
        layers = list(self.hidden_layers)
        layers[layer] = value
        self.hidden_layers = layers
        self._on_structure_changed()

    def _on_live_changed(self) -> None:
        """Learning rate, regularisation and batch size take effect without a reset."""
        s = self.session
        if s is None:
            return
        s.config.learning_rate = LEARNING_RATES[self.lr_index]
        s.config.regularization = REGULARIZATION_KEYS[self.reg_index]
        s.config.reg_rate = REGULARIZATION_RATES[self.reg_rate_index]
        s.config.batch_size = self.batch_size

    def _on_axes_changed(self) -> None:
        if self.axis_x == self.axis_y and self.extracted is not None and len(self.extracted.attributes) > 1:
            # keep the two axes different: move the other one out of the way
            self.axis_y = (self.axis_x + 1) % len(self.extracted.attributes)
            self.axis_y_combo.setCurrentIndex(self.axis_y)
        self._refresh()

    # ----- training ---------------------------------------------------------------------------

    def _reset(self) -> None:
        self._pause()
        self.init_seed += 1
        self._build_session()

    def _toggle_play(self) -> None:
        if self.playing:
            self._pause()
            self._send_outputs()
        elif self.session is not None:
            self.Warning.diverged.clear()
            self.playing = True
            self.play_button.setText("Pause")
            self.timer.start()

    def _pause(self) -> None:
        self.playing = False
        self.timer.stop()
        self.play_button.setText("Play")

    def _step_once(self) -> None:
        if self.session is None:
            return
        self._pause()
        self.session.step(1)
        self._after_training()
        self._send_outputs()

    def _tick(self) -> None:
        if self.session is None:
            self._pause()
            return
        self.session.step(self.epochs_per_update, budget=TICK_BUDGET)
        self._after_training()
        if self.playing and not np.isfinite(self.session.train_loss):
            self._pause()
            self.Warning.diverged()
            self._send_outputs()

    def _after_training(self) -> None:
        self._refresh()

    # ----- drawing ----------------------------------------------------------------------------

    def _clear_views(self) -> None:
        self.network_view.set_state([], [], [], [], [], [])
        self.output_view.image = None
        self.output_view.set_state(
            self.output_view.image, (0.0, 1.0, 0.0, 1.0), np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3)),
            self.show_test, ("", ""),
        )
        self.loss_view.set_history([])
        self.stats.setText("")

    def _refresh(self) -> None:
        s, info = self.session, self.extracted
        if s is None or info is None:
            return
        ax, ay = self.axis_x, self.axis_y
        acts = s.grid_activations(ax, ay, GRID)
        images = [[to_qimage(diverging(acts[0][:, :, k])) for k in range(acts[0].shape[2])]]
        for layer in acts[1:-1]:
            images.append([to_qimage(diverging(layer[:, :, k])) for k in range(layer.shape[2])])
        out = acts[-1]
        if s.head == "softmax":
            images.append([to_qimage(white_to(out[:, :, k], k)) for k in range(out.shape[2])])
        else:
            images.append([to_qimage(output_image(out, s.head, self.value_scale))])
        self.network_view.set_state(
            s.features.labels, s.network.sizes, s.network.weights, s.network.biases, images, s.class_names if s.head == "softmax" else []
        )

        x0, x1 = s.axis_range(ax)
        y0, y1 = s.axis_range(ay)
        names = (s.base_names[ax], s.base_names[ay])
        train_xy, test_xy = s.raw[s.train_idx][:, [ax, ay]], s.raw[s.test_idx][:, [ax, ay]]
        self.output_view.set_state(
            to_qimage(output_image(out, s.head, self.value_scale)),
            (x0, x1, y0, y1),
            train_xy,
            point_colors(s.y_train, s.head, self.value_scale),
            test_xy,
            point_colors(s.y_test, s.head, self.value_scale),
            self.show_test,
            names,
        )
        self.loss_view.set_history(s.history)
        self.stats.setText(self._stats_text(s))

    @staticmethod
    def _stats_text(s: PlaygroundSession) -> str:
        parts = [f"<b>Epoch {s.epoch:,}</b>", f"train loss {s.train_loss:.3f}", f"test loss {s.test_loss:.3f}"]
        if s.task == "classification":
            parts += [f"train accuracy {s.train_accuracy:.0%}", f"test accuracy {s.test_accuracy:.0%}"]
        return " &nbsp;|&nbsp; ".join(parts)

    # ----- outputs ----------------------------------------------------------------------------

    def _send_outputs(self) -> None:
        s, info = self.session, self.extracted
        if s is None or info is None:
            self.Outputs.model.send(None)
            self.Outputs.data.send(None)
            self.Outputs.hidden.send(None)
            return
        self.Outputs.model.send(snapshot_model(s, info))
        self.Outputs.data.send(data_with_subset(s, info))
        self.Outputs.hidden.send(hidden_output(s, info))

    def onDeleteWidget(self) -> None:  # noqa: N802
        self.timer.stop()
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWNeuralPlayground).run()
