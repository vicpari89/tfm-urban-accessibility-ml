"""Model builders: one module per model, each with build_model(seed) -> an UNTRAINED estimator."""

from sklearn.base import BaseEstimator

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling.models import (
    model_dummy,
    model_logistic_regression,
)

BUILDERS = {
    "dummy": model_dummy.build_model,
    "logistic_regression": model_logistic_regression.build_model,
}


def build_model(name: str, seed: int = SEED) -> BaseEstimator:
    """Untrained model by name (a key of BUILDERS)."""
    if name not in BUILDERS:
        raise ValueError(f"Unknown model: {name}. Options: {list(BUILDERS)}")
    return BUILDERS[name](seed)
