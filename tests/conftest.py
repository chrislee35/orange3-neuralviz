"""Make ``orangecontrib.neuralviz`` importable from a source checkout.

``orangecontrib`` is a namespace shared by every Orange add-on, and the copy installed next to
Orange wins over a checkout on ``sys.path``, so the checkout is appended to its search path.
"""

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# The tests assert on English text, so keep the developer's own Orange language setting (read from the user config dir) out of them.
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="nv-test-config-")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import orangecontrib  # noqa: E402

path = os.path.join(ROOT, "orangecontrib")
if path not in orangecontrib.__path__:
    orangecontrib.__path__.insert(0, path)
