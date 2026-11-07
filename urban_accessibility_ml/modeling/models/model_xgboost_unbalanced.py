"""XGBoost WITHOUT balanced classes: the same model as model_xgboost_balanced, trained without sample weights.

Only used in notebook 0.09, to see what the class balancing of model_xgboost_balanced changes.
"""

from xgboost import XGBClassifier

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling.models import model_xgboost_balanced


def build_model(seed: int = SEED) -> XGBClassifier:
    """Return a plain XGBClassifier, with the same defaults as model_xgboost_balanced.

    Input:  seed (int)
    Output: an UNTRAINED XGBClassifier (it does not balance the classes)
    """
    return XGBClassifier(
        n_estimators=model_xgboost_balanced.N_ESTIMATORS,
        random_state=seed,
        n_jobs=model_xgboost_balanced.N_JOBS,
    )
