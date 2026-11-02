"""Model training: one experiment on one split.

An experiment is a combination of:
- split: which rows train and which test -> preprocessing/split.py
- feature set and profile -> preprocessing/features.py
- model -> modeling/models/...
"""

import pandas as pd

from urban_accessibility_ml.modeling.evaluate import evaluate
from urban_accessibility_ml.modeling.models import build_model
from urban_accessibility_ml.preprocessing.features import get_xy
from urban_accessibility_ml.preprocessing.split import Split


def run_one(
    df: pd.DataFrame,
    split: Split,
    feature_set: str,
    profile: str,
    model_name: str,
) -> dict:
    """
    Train and evaluate one experiment on one split.

    :param df: unified dataset.
    :param split: one train/test partition.
    :param feature_set: which dimensions set use: one of config.FEATURE_SETS.
    :param profile: which profile: one of config.PROFILES.
    :param model_name: model to train: one of the keys of modeling.models.BUILDERS.
    :return: one result row, a dict with the experiment (strategy, name, fold, feature_set,
        profile, model), the size of each set (n_train, n_test) and one value per metric of
        config.METRICS.
    """
    # 1. X and y, for all the rows
    X, y = get_xy(df, feature_set, profile)

    # 2. Rows of train/test
    X_train, y_train = X.loc[split.train_ids], y.loc[split.train_ids]
    X_test, y_test = X.loc[split.test_ids], y.loc[split.test_ids]

    # 3. Train
    model = build_model(model_name)
    model.fit(X_train, y_train)

    # 4. Predict
    y_pred = model.predict(X_test)

    # 5. Evaluate: {metric name: value}, one per config.METRICS
    metrics = evaluate(y_test, y_pred)

    # 6. Return the experiment, the size of each set and the metrics
    return {
        "strategy": split.strategy,
        "name": split.name,
        "fold": split.fold,
        "feature_set": feature_set,
        "profile": profile,
        "model": model_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        **metrics,
    }
