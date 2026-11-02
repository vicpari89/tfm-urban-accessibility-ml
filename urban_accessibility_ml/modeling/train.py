"""Model training.

For now (task #6) only the baseline. The rest of the models (random forest, XGBoost, LightGBM)
and the loop over the experiment matrix come in task #7.
"""

import pandas as pd
from sklearn.base import BaseEstimator

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.preprocessing.split import Split


def build_baseline_models(seed: int = SEED) -> dict[str, BaseEstimator]:
    """Reference models, by name.

    - 'dummy': always predicts the most frequent class (the floor any model must beat)
    - 'logistic_regression': the reference model chosen in the memoria, with class weights
      to compensate for the rare class A (class_weight="balanced") and a fixed seed
    """
    raise NotImplementedError


def run_one(
    df: pd.DataFrame,
    split: Split,
    feature_set: str,
    profile: str,
    model_name: str,
) -> dict:
    """Train and evaluate ONE experiment on ONE split. Return ONE result row (a dict).

    Steps:
      1. X, y = get_xy(df, feature_set, profile)       (features.py)
      2. train / test rows = X.loc[split.train_ids], X.loc[split.test_ids]  (same for y)
      3. build the model, fit on train, predict on test
      4. metrics = evaluate(y_test, y_pred)             (evaluate.py)
      5. return {strategy, name, fold, feature_set, profile, model, n_train, n_test, **metrics}

    This is the unit that task #7 will call inside its loop.
    """
    raise NotImplementedError
