import sys
import warnings

from green_mbtools import pesto

# NB: FutureWarning (not DeprecationWarning) so the notice is shown by
# default in every import context. Python's default filters suppress
# DeprecationWarning unless it is triggered from `__main__`, which would
# hide it whenever `mbanalysis` is imported from inside another module.
warnings.warn(
    "`mbanalysis` is deprecated and will be removed in green-mbtools v1.1; "
    "import `green_mbtools.pesto` instead.",
    FutureWarning,
    stacklevel=2,
)

sys.modules['mbanalysis'] = pesto
