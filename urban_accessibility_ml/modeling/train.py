"""Model training: one experiment on one split.

An experiment is a combination of:
- split: which rows train and which test -> preprocessing/split.py
- feature set and profile -> preprocessing/features.py
- model -> modeling/models/...

Flow: preprocessing/split.build_all_splits -> experiments.run_experiments (calls run_one for each
combination) -> experiments.save_results
"""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_sample_weight

from urban_accessibility_ml.modeling.evaluate import evaluate
from urban_accessibility_ml.modeling.models import NEED_SAMPLE_WEIGHT, build_model
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
    model = _train(model_name, X_train, y_train)

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


def _train(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    """
    Build the model and train it on the train rows.

    :param model_name: one of the keys of modeling.models.BUILDERS.
    :param X_train: inputs of the train rows.
    :param y_train: encoded classes of the train rows.
    :return: the trained model.
    """
    model = build_model(model_name)
    model.fit(X_train, y_train, **_fit_params(model_name, y_train))
    return model


def _fit_params(model_name: str, y_train: pd.Series) -> dict:
    """
    Extra arguments for fit. Some models (XGBoost) have no class_weight: their classes are
    balanced with one weight per train row (sample_weight).
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
    https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn

    :param model_name: one of the keys of modeling.models.NEED_SAMPLE_WEIGHT.
    :param y_train: encoded classes of the train rows.
    :return: {"sample_weight": weights} if the model needs it, an empty dict if not.
    """
    if NEED_SAMPLE_WEIGHT[model_name]:
        return {"sample_weight": compute_sample_weight("balanced", y_train)}
    return {}
