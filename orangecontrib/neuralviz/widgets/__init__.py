"""
Neural Viz Widgets
==================

Widgets for building, training and watching small neural networks.
"""

import sysconfig

NAME = "Neural Viz"
DESCRIPTION = "Build, train and watch small neural networks"

ICON = "icons/nv-playground.svg"
PRIORITY = 1010
BACKGROUND = "#ffcc99"

WIDGET_HELP_PATH = (
    # Used for development. Build the help pages first with `make docs`.
    ("{DEVELOP_ROOT}/doc/_build/html/index.html", None),
    # Documentation included in a wheel.
    ("{}/help/orange3-neuralviz/index.html".format(sysconfig.get_path("data")), None),
    # Online documentation.
    ("https://chrislee35.github.io/orange3-neuralviz/", ""),
)
