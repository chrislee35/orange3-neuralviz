# Changelog

## 0.0.1 (2026-11-22)

### Added
- **Neural Playground** widget: a TensorFlow-Playground-style network builder. Choose input features (the data's numeric columns plus x², y², x·y, sin x, sin y of the first two), 0-6 hidden layers of 1-8 neurons, the activation (ReLU, tanh, sigmoid, linear), learning rate, L1/L2 regularization, batch size and train/test split. Reset, Play/Pause and Step; learning rate, regularization and batch size can be changed while it trains. The diagram shows a heat map of every neuron's output and edges coloured and sized by weight (hover for the value); next to it are the decision map over the data, test points with a dark outline, and the train/test loss curve. Works with binary and multi-class classification and with regression. Outputs the trained *Model*, the *Data* with a train/test *Subset* column, and the *Hidden Layer Output* of the last hidden layer.
- **Playground Data** widget: circle, XOR, two Gaussians and spiral (classification), plane and multiple Gaussians (regression), with noise, size and seed.
- Qt-free numpy engine (network, features, data sets, training session) with unit tests.
