# orange3-neuralviz

[Orange3](https://orangedatamining.com) widgets for building, training and *watching* small neural networks, in the spirit of the
[TensorFlow Playground](https://playground.tensorflow.org/). Pick some inputs, add hidden layers and neurons, press **Play**, and see
what every neuron learns while the loss falls.

Unlike the browser original, it works on **your own data**: connect any table with numeric features and a class (or numeric target) and
get a trained model back for Orange's *Predictions*, *Test and Score* and other widgets.

## Widgets

| Widget | What it does |
|--------|--------------|
| **Neural Playground** | The playground itself: network diagram with a live heat map in every neuron, decision boundary over your data, train/test loss curve. Outputs the trained *Model*, the *Data* (tagged train or test) and the *Hidden Layer Output*. |
| **Playground Data** | Generates the Playground's toy data sets (circle, XOR, two Gaussians, spiral; plane and multiple Gaussians for regression) with adjustable noise. |

*Neural Playground* works without any input (it has the same built-in data sets), so you can drop it on the canvas and press Play.

## Installation

### From Orange

Within the Add-ons installer, click on "Add more..." and type in `orange3-neuralviz`.

### With pip

Install into the same Python environment as Orange:

```
pip install orange3-neuralviz
```

Requirements:

- Python 3.11, 3.12 or 3.13
- Orange3 3.41.0 or newer (Linux, Windows and macOS)
- numpy 2.0 or newer

The networks are trained with a small built-in numpy implementation, so there is **no TensorFlow or PyTorch dependency** and no
download of models or data.

### From source

```
git clone https://github.com/chrislee35/orange3-neuralviz
cd orange3-neuralviz
pip install -e .
```

## Documentation

Online at <https://chrislee35.github.io/orange3-neuralviz/>; build it locally with `make docs`.

## Development

```
pip install -e ".[test]"
make test
```

The engine (`orangecontrib/neuralviz/util/network.py`, `session.py`, `features.py`, `datasets.py`) is plain numpy with no Qt or Orange
imports, and is unit tested, including a numerical gradient check of backpropagation.

## License

CC-BY-NC-SA-4.0, see [LICENSE](LICENSE).
