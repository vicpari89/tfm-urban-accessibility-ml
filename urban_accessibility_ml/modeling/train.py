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

from urban_accessibility_ml.modeling.evaluate import evaluate
from urban_accessibility_ml.modeling.models import build_model
from urban_accessibility_ml.modeling.tuning.tuning_constants import SEARCH_SPACES
from urban_accessibility_ml.modeling.tuning.tuning_results import tuning_key
from urban_accessibility_ml.preprocessing.features import get_xy
from urban_accessibility_ml.preprocessing.split import Split


def run_one(
    df: pd.DataFrame,
    split: Split,
    feature_set: str,
    profile: str,
    model_name: str,
    hyperparameters: dict | None = None,
) -> dict:
    """
    Train and evaluate one experiment on one split.

    :param df: unified dataset.
    :param split: one train/test partition.
    :param feature_set: which dimensions set use: one of config.FEATURE_SETS.
    :param profile: which profile: one of config.PROFILES.
    :param model_name: model to train: one of the keys of modeling.models.BUILDERS.
    :param hyperparameters: {tuning_key: params} (best_params.json). None: the model defaults.
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
    model = _train(model_name, X_train, y_train, _params_for(hyperparameters, split, feature_set, model_name))

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


def _params_for(
    hyperparameters: dict | None, split: Split, feature_set: str, model_name: str
) -> dict | None:
    """
    The tuned hyperparameters of a model in a split and feature set.

    :return: None if there are no hyperparameters or the model is not tuned (dummy).
    :raises KeyError: if a tuned model has no entry.
    """
    if hyperparameters is None or model_name not in SEARCH_SPACES:
        return None
    key = tuning_key(split, feature_set, model_name)
    if key not in hyperparameters:
        raise KeyError(f"No hyperparameters for {key}")
    return hyperparameters[key]


def _train(
    model_name: str, X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None
) -> BaseEstimator:
    """
    Build the model and train it on the train rows.

    :param model_name: one of the keys of modeling.models.BUILDERS.
    :param X_train: inputs of the train rows.
    :param y_train: encoded classes of the train rows.
    :param params: hyperparameters of the model. None: the default ones.
    :return: the trained model.
    """
    model = build_model(model_name, params=params)
    model.fit(X_train, y_train)
    return model
