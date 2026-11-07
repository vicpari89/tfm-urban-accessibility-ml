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

MODEL_MODULES = {
    "dummy": model_dummy,
    "logistic_regression": model_logistic_regression,
    "random_forest": model_random_forest,
    "xgboost_balanced": model_xgboost_balanced,
    "xgboost_unbalanced": model_xgboost_unbalanced,
    "lightgbm": model_lightgbm,
}

BUILDERS = {name: module.build_model for name, module in MODEL_MODULES.items()}

# Models that must be trained with balanced sample weights (see run_one in train.py)
NEED_SAMPLE_WEIGHT = {name: module.NEED_SAMPLE_WEIGHT for name, module in MODEL_MODULES.items()}


def build_model(name: str, seed: int = SEED, params: dict | None = None) -> BaseEstimator:
    """
    Untrained model by name (a key of BUILDERS).

    :param params: hyperparameters, passed by name to the builder of the model. A name the
        builder does not accept (or any, for dummy and xgboost_unbalanced) raises TypeError.
    """
    if name not in BUILDERS:
        raise ValueError(f"Unknown model: {name}. Options: {list(BUILDERS)}")
    return BUILDERS[name](seed, **(params or {}))
