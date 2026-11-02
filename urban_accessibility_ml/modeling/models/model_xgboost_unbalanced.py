"""XGBoost WITHOUT balanced classes: the same model as model_xgboost_balanced, trained without sample weights.

Only used in notebook 0.09, to see what the class balancing of model_xgboost_balanced changes.
"""

from xgboost import XGBClassifier

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling.models import model_xgboost_balanced

NEED_SAMPLE_WEIGHT = False  # True: run_one trains with balanced sample_weight


def build_model(seed: int = SEED) -> XGBClassifier:
    """Return an XGBoost classifier, the same as model_xgboost_balanced.

    Input:  seed (int)
    Output: an UNTRAINED XGBClassifier
    """
    return model_xgboost_balanced.build_model(seed)
