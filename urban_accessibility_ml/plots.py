"""Style and helpers of the figures: the visual system of the thesis (tfm.mplstyle and paletas.py)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from urban_accessibility_ml.config import FIGURES_DIR
from urban_accessibility_ml.paletas import ACC, CMAP_SEC, SERIES  # noqa: F401 (used by the notebooks)

STYLE_FILE = Path(__file__).parent / "tfm.mplstyle"
# Accessibility classes with the colours of the design system (blue = accessible, orange = not accessible)
CLASS_COLORS = {"A": ACC["Accesible"], "B": ACC["Deficiente"], "C": ACC["No accesible"]}
# Continuous scales: accessibility (orange = low, blue = high), and divergent around 0 for correlations
CMAP_ACC = LinearSegmentedColormap.from_list(
    "tfm_acc", [ACC["No accesible"], ACC["Deficiente"], ACC["Practicable"], ACC["Accesible"]]
)
CMAP_DIV = LinearSegmentedColormap.from_list("tfm_div", [ACC["No accesible"], "#FFFFFF", ACC["Accesible"]])


def sequential_colors(n: int) -> list:
    """n colours of the sequential scale, from light (worst class) to dark (best class)."""
    return [tuple(c) for c in CMAP_SEC(np.linspace(0.25, 1, n))]


def use_style():
    """Apply the visual system of the thesis to every figure of the notebook."""
    plt.style.use(STYLE_FILE)


def save_figure(name: str, folder: str, formats=("png",), fig=None):
    """Save a figure in reports/figures/<folder>/ to use it in the thesis (PNG)."""
    path = FIGURES_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    for fmt in formats:
        fig.savefig(path / f"{name}.{fmt}", dpi=200, bbox_inches="tight")
