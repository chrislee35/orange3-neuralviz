# Playground Data

Generate the small two-dimensional data sets of the TensorFlow Playground.

**Outputs**

- Data: a table with two numeric features, `x1` and `x2`, and a class (`class`, values `0` and `1`) or a numeric target (`value`)

**Data sets**

- **Circle**: class 1 inside, class 0 in a ring around it.
- **XOR**: class 1 where `x1` and `x2` have the same sign.
- **Two Gaussians**: two blobs, centred at (2, 2) and (-2, -2).
- **Spiral**: two interleaved spirals. The hardest one: it needs more neurons, more layers or more input features such as sin(x1) and sin(x2).
- **Plane** (regression): the target grows linearly with `x1 + x2`.
- **Multiple Gaussians** (regression): the target is a sum of four positive and negative bumps.

**Settings**

- **Samples**: the number of rows.
- **Noise**: jitter added to the coordinates, as a percentage of the data's radius. `0` gives clean data and `50` heavily overlapping classes.
- **Random seed** and **New random data**: the same seed always gives the same data.

The data sets are the same as the ones *Neural Playground* generates for itself, so use this widget when you want to look at the data in a *Data Table* or *Scatter Plot*, or feed the same data to other learners to compare them.
