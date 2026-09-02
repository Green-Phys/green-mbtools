import sys
import importlib

import pytest

from green_mbtools import pesto


def test_mbanalysis_alias_warns_and_points_to_pesto():
    """`import mbanalysis` still works during the 1.x series but must emit a
    FutureWarning and alias `green_mbtools.pesto`. Removed in v1.1.

    FutureWarning (rather than DeprecationWarning) is used deliberately so the
    notice is shown by default even when `mbanalysis` is imported from inside
    another module, not just from `__main__`.
    """
    # Ensure a fresh import so the package __init__ (and its warning) runs.
    sys.modules.pop('mbanalysis', None)

    with pytest.warns(FutureWarning, match="green_mbtools.pesto"):
        mbanalysis = importlib.import_module('mbanalysis')

    assert mbanalysis is pesto
