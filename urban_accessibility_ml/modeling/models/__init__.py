"""Model builders: one module per model, each with build_model(seed) -> an UNTRAINED estimator."""

from sklearn.base import BaseEstimator

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling.models import (
    model_dummy,
    model_lightgbm,
    model_logistic_regression,
    model_random_forest,
    model_xgboost_balanced,
    model_xgboost_unbalanced,
)

BUILDERS = {
    "dummy": model_dummy.build_model,
    "logistic_regression": model_logistic_regression.build_model,
    "random_forest": model_random_forest.build_model,
    "xgboost_balanced": model_xgboost_balanced.build_model,
    "xgboost_unbalanced": model_xgboost_unbalanced.build_model,
    "lightgbm": model_lightgbm.build_model,
}

# Models that must be trained with balanced sample weights (see run_one in train.py)
NEED_SAMPLE_WEIGHT = {
    "dummy": model_dummy.NEED_SAMPLE_WEIGHT,
    "logistic_regression": model_logistic_regression.NEED_SAMPLE_WEIGHT,
    "random_forest": model_random_forest.NEED_SAMPLE_WEIGHT,
    "xgboost_balanced": model_xgboost_balanced.NEED_SAMPLE_WEIGHT,
    "xgboost_unbalanced": model_xgboost_unbalanced.NEED_SAMPLE_WEIGHT,
    "lightgbm": model_lightgbm.NEED_SAMPLE_WEIGHT,
}


def build_model(name: str, seed: int = SEED) -> BaseEstimator:
    """Untrained model by name (a key of BUILDERS)."""
    if name not in BUILDERS:
        raise ValueError(f"Unknown model: {name}. Options: {list(BUILDERS)}")
    return BUILDERS[name](seed)
