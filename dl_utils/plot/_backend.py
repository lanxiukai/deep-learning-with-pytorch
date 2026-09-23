"""Select the repository's non-interactive Matplotlib backend."""

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot

__all__ = ["matplotlib", "pyplot"]
