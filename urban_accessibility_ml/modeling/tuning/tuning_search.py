"""Hyperparameter tuning, inside a train/test partition and ONLY with its train rows.

To choose the values, the train rows are split again into 3 groups (StratifiedGroupKFold). A street
always stays in the same group, and each group keeps a similar share of every class. All the
candidates of a model are tested on the same groups.

The search itself is done with GridSearchCV (kind "grid") or RandomizedSearchCV (kind "random").

Settings and search spaces: modeling/tuning/tuning_constants.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from tqdm.auto import tqdm

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling.evaluate import METRIC_FUNCTIONS
from urban_accessibility_ml.modeling.experiments import _index_by_key
from urban_accessibility_ml.modeling.models import build_model
from urban_accessibility_ml.modeling.tuning.tuning_constants import (
    SEARCH_SPACES,
    TUNING_FEATURE_SET,
    TUNING_INNER_SPLITS,
    TUNING_METRIC,
    TUNING_N_CANDIDATES,
    TUNING_PROFILE,
    TUNING_TIE_TOLERANCE,
)
from urban_accessibility_ml.preprocessing.features import get_xy
from urban_accessibility_ml.preprocessing.split import STREET_COL, Split

# How complex is a candidate, lower is simpler (tuples are compared element by element).
# Used to break the ties.
COMPLEXITY = {
    # smaller C = stronger regularization; at the same C, l1 is simpler (sets weights to 0)
    "logistic_regression": lambda p: (p["C"], 0 if p["penalty"] == "l1" else 1),
    # simpler trees, then bigger leaves, then fewer variables per split
    "random_forest": lambda p: (p["max_depth"], -p["min_samples_leaf"], p["max_features"]),
    # fewer leaves, then fewer trees, then bigger leaves (more min_child_samples)
    "lightgbm": lambda p: (p["num_leaves"], p["n_estimators"], -p["min_child_samples"]),
    # simpler trees, then fewer trees, then a bigger min_child_weight (more conservative)
    "xgboost_balanced": lambda p: (p["max_depth"], p["n_estimators"], -p["min_child_weight"]),
}


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """
    Choose the best hyperparameters of one model, using only the train rows of one partition.

    :param model_name: a key of SEARCH_SPACES.
    :param X_train: inputs of the train rows.
    :param y_train: encoded classes of the train rows.
    :param groups_train: street of each train row, to keep a street in a single inner fold.
    :return: (best_params, candidates). candidates has one row per candidate, with its params, the
        mean and std of the metric over the inner folds, and selected (True for the best one).
        Candidates within TUNING_TIE_TOLERANCE of the best mean are tied: the simplest wins.
    :raises ValueError: if the model has no search space.
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space for: {model_name}. Options: {list(SEARCH_SPACES)}")

    # 1. Inner folds: https://nbviewer.org/github/Yorko/mlcourse.ai/blob/main/jupyter_english/tutorials/nested_cross_validation_tatyana_kudasova.ipynb
    cv = StratifiedGroupKFold(n_splits=TUNING_INNER_SPLITS, shuffle=True, random_state=SEED)

    # 2. Score every candidate on every inner fold
    search = _build_search(model_name, cv)
    search.fit(X_train, y_train, groups=groups_train)

    # 3. All the candidates, with the best one marked
    results = search.cv_results_
    candidates = pd.DataFrame(
        {
            "params": results["params"],
            f"{TUNING_METRIC}_mean": results["mean_test_score"],
            f"{TUNING_METRIC}_std": results["std_test_score"],
        }
    )
    candidates["selected"] = candidates.index == search.best_index_
    return search.best_params_, candidates


def tune_partition(df: pd.DataFrame, split: Split, feature_set: str) -> tuple[dict, pd.DataFrame]:
    """
    Tune every model of SEARCH_SPACES with the TRAIN rows of one partition and one feature set
    (with the TUNING_PROFILE as reference). The test rows are never read.

    :param df: unified dataset, indexed by key_tramo.
    :param split: one train/test partition.
    :param feature_set: one of config.FEATURE_SETS.
    :return: (best_params, candidates). best_params is {model: params}. candidates joins the
        candidates of every model, with the columns strategy, name, fold and model.
    """
    X, y = get_xy(df, feature_set, TUNING_PROFILE)
    X_train, y_train = X.loc[split.train_ids], y.loc[split.train_ids]
    groups_train = df.loc[split.train_ids, STREET_COL]

    best_params, tables = {}, []
    for model_name in SEARCH_SPACES:
        best_params[model_name], table = tune_model(model_name, X_train, y_train, groups_train)
        tables.append(
            table.assign(
                strategy=split.strategy, name=split.name, fold=split.fold, model=model_name
            )
        )
    return best_params, pd.concat(tables, ignore_index=True)


def tune_all(df: pd.DataFrame, splits: list[Split]) -> tuple[dict, pd.DataFrame]:
    """
    tune_partition for every partition.

    :param df: unified dataset, with key_tramo as the index or as a column.
    :param splits: partitions to tune (build_all_splits).
    :return: (best_params, candidates). best_params is {Split.key: {model: params}}; candidates
        joins the tables of all the partitions.
    """
    indexed = _index_by_key(df)
    best_params, tables = {}, []
    for split in tqdm(splits, desc="tuning"):
        best_params[split.key], table = tune_partition(indexed, split, TUNING_FEATURE_SET)
        tables.append(table)
    return best_params, pd.concat(tables, ignore_index=True)


def _build_search(model_name: str, cv: StratifiedGroupKFold) -> GridSearchCV | RandomizedSearchCV:
    """
    The search of a model, ready to fit: a grid or a random one, as its search space says.
    The refit function applies the tie rule.
    """
    spec = SEARCH_SPACES[model_name]
    options = {
        "scoring": make_scorer(METRIC_FUNCTIONS[TUNING_METRIC]),
        "cv": cv,
        "refit": lambda cv_results: _pick_best(model_name, cv_results),
    }
    if spec["kind"] == "grid":
        return GridSearchCV(build_model(model_name), spec["space"], **options)
    return RandomizedSearchCV(
        build_model(model_name),
        spec["space"],
        n_iter=TUNING_N_CANDIDATES,
        random_state=SEED,
        **options,
    )


def _pick_best(model_name: str, cv_results: dict) -> int:
    """Index of the simplest candidate among those within the tolerance of the best mean."""
    means = np.asarray(cv_results["mean_test_score"])
    tied = np.flatnonzero(means >= means.max() - TUNING_TIE_TOLERANCE)
    return min(tied, key=lambda i: COMPLEXITY[model_name](cv_results["params"][i]))
