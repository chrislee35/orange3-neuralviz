from AnyQt.QtCore import Qt
from Orange.data import Table
from Orange.widgets import gui, widget
from Orange.widgets.widget import Output

from orangecontrib.neuralviz.util.datasets import DATASETS
from orangecontrib.neuralviz.util.orange_io import generated_table
from orangecontrib.neuralviz.util.translate import _tr
from orangecontrib.neuralviz.util.typed_settings import setting

DATASET_KEYS = list(DATASETS)


class OWPlaygroundData(widget.OWWidget):
    name = _tr.m[40, "Playground Data"]
    description = _tr.m[41, "Generate the small two-dimensional data sets of the TensorFlow Playground."]
    icon = "icons/nv-data.svg"
    priority = 20
    keywords = ["circle", "xor", "spiral", "gaussian", "synthetic", "generator", "toy"]
    want_main_area = False
    resizing_enabled = False

    class Outputs:
        data = Output("Data", Table)

    dataset_index: int = setting(0)
    n_samples: int = setting(200)
    noise: int = setting(0)
    seed: int = setting(0)

    def __init__(self) -> None:
        super().__init__()
        box = gui.vBox(self.controlArea, _tr.m[42, "Data set"])
        gui.comboBox(
            box, self, "dataset_index", items=list(DATASETS.values()), orientation=Qt.Orientation.Horizontal, callback=self.generate
        )
        gui.spin(box, self, "n_samples", 20, 2000, step=20, label=_tr.m[26, "Samples:"], callback=self.generate)
        gui.hSlider(
            box,
            self,
            "noise",
            label=_tr.m[27, "Noise:"],
            minValue=0,
            maxValue=50,
            step=1,
            intOnly=True,
            labelFormat=" %d%%",
            callback=self.generate,
        )
        gui.spin(box, self, "seed", 0, 9999, label=_tr.m[43, "Random seed:"], callback=self.generate)
        gui.button(box, self, _tr.m[28, "New random data"], callback=self.reseed)
        self.generate()

    def reseed(self) -> None:
        self.seed = (self.seed + 1) % 10000
        self.generate()

    def generate(self) -> None:
        kind = DATASET_KEYS[self.dataset_index]
        self.Outputs.data.send(generated_table(kind, self.n_samples, self.noise, self.seed))


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWPlaygroundData).run()
