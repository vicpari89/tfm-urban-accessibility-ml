"""Hyperparameter tuning, inside a train/test partition and ONLY with its train rows.

To choose the values, the train rows are split again into 3 groups (StratifiedGroupKFold). A street
always stays in the same group, and each group keeps a similar share of every class. All the
candidates of a model are tested on the same groups.

The search itself is done with GridSearchCV (kind "grid") or RandomizedSearchCV (kind "random").

Settings and search spaces: modeling/tuning/tuning_constants.py.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling.evaluate import METRIC_FUNCTIONS
from urban_accessibility_ml.modeling.models import build_model
from urban_accessibility_ml.modeling.tuning.tuning_constants import (
    SEARCH_SPACES,
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
    n_jobs: int = 1,
    seed: int | None = None,
    n_candidates: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Choose the best hyperparameters of one model, using only the train rows of one partition.

    :param model_name: a key of SEARCH_SPACES.
    :param X_train: inputs of the train rows.
    :param y_train: encoded classes of the train rows.
    :param groups_train: street of each train row, to keep a street in a single inner fold.
    :param n_jobs: number of parallel runs.
    :param seed: seed of the inner folds and of the random search. By default config.SEED.
    :param n_candidates: candidates of a random search. By default TUNING_N_CANDIDATES.
    :return: (best_params, candidates). candidates has one row per candidate, with its params, the
        mean and std of the metric over the inner folds, and selected (True for the best one).
        Candidates within TUNING_TIE_TOLERANCE of the best mean are tied: the simplest wins.
    :raises ValueError: if the model has no search space.
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space for: {model_name}. Options: {list(SEARCH_SPACES)}")

    # 1. Inner folds: https://nbviewer.org/github/Yorko/mlcourse.ai/blob/main/jupyter_english/tutorials/nested_cross_validation_tatyana_kudasova.ipynb
    seed = SEED if seed is None else seed
    cv = StratifiedGroupKFold(n_splits=TUNING_INNER_SPLITS, shuffle=True, random_state=seed)

    # 2. Score every candidate on every inner fold
    search = _build_search(model_name, cv, n_jobs, seed, n_candidates)
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


def tune_partition(
    df: pd.DataFrame,
    split: Split,
    feature_set: str,
    n_jobs: int = 1,
    seed: int | None = None,
    n_candidates: int | None = None,
    models: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Tune every model of SEARCH_SPACES with the TRAIN rows of one partition and feature set.
    The test rows are never read.

    :param df: unified dataset, indexed by key_tramo.
    :param split: one train/test partition.
    :param feature_set: one of config.FEATURE_SETS.
    :param n_jobs: number of parallel runs.
    :param seed: see tune_model.
    :param n_candidates: see tune_model.
    :param models: models to tune. By default all of SEARCH_SPACES.
    :return: (best_params, candidates). best_params is {model: params}. candidates joins the
        candidates of every model, with the columns strategy, name, fold and model.
    """
    X_train, y_train, groups_train = train_data(df, split, feature_set)

    best_params, tables = {}, []
    for model_name in SEARCH_SPACES if models is None else models:
        best_params[model_name], table = tune_model(
            model_name, X_train, y_train, groups_train, n_jobs, seed, n_candidates
        )
        tables.append(
            table.assign(
                strategy=split.strategy, name=split.name, fold=split.fold, model=model_name
            )
        )
    return best_params, pd.concat(tables, ignore_index=True)


def train_data(df: pd.DataFrame, split: Split, feature_set: str):
    """X_train, y_train and the street of each train row (df indexed by key_tramo)."""
    X, y = get_xy(df, feature_set, TUNING_PROFILE)
    return X.loc[split.train_ids], y.loc[split.train_ids], df.loc[split.train_ids, STREET_COL]


def time_candidate(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_jobs: int = 1,
    seed: int | None = None,
    n_candidates: int | None = None,
) -> tuple[float, int]:
    """
    Seconds that one candidate takes the inner folds (mean of 3 candidates spread over the
    search space), and the number of candidates of the search.
    """
    seed = SEED if seed is None else seed
    n_candidates = TUNING_N_CANDIDATES if n_candidates is None else n_candidates
    spec = SEARCH_SPACES[model_name]
    if spec["kind"] == "grid":
        candidates = list(ParameterGrid(spec["space"]))
    else:  # the same ones that RandomizedSearchCV samples
        candidates = list(ParameterSampler(spec["space"], n_candidates, random_state=seed))
    cv = StratifiedGroupKFold(n_splits=TUNING_INNER_SPLITS, shuffle=True, random_state=seed)
    folds = list(cv.split(X_train, y_train, groups_train))

    chosen = candidates[:: max(1, len(candidates) // 3)][:3]
    start = time.perf_counter()
    for params in chosen:
        for train, _ in folds:
            _estimator(model_name, params, n_jobs).fit(X_train.iloc[train], y_train.iloc[train])
    return (time.perf_counter() - start) / len(chosen), len(candidates)


def _build_search(
    model_name: str,
    cv: StratifiedGroupKFold,
    n_jobs: int,
    seed: int | None = None,
    n_candidates: int | None = None,
) -> GridSearchCV | RandomizedSearchCV:
    """
    The search of a model, ready to fit: a grid or a random one, as its search space says.
    The refit function applies the tie rule.
    """
    spec = SEARCH_SPACES[model_name]
    options = {
        "scoring": make_scorer(METRIC_FUNCTIONS[TUNING_METRIC]),
        "cv": cv,
        "n_jobs": n_jobs,
        "refit": lambda cv_results: _pick_best(model_name, cv_results),
    }
    if spec["kind"] == "grid":
        return GridSearchCV(_estimator(model_name, None, n_jobs), spec["space"], **options)
    return RandomizedSearchCV(
        _estimator(model_name, None, n_jobs),
        spec["space"],
        n_iter=TUNING_N_CANDIDATES if n_candidates is None else n_candidates,
        random_state=SEED if seed is None else seed,
        **options,
    )


def _estimator(model_name: str, params: dict | None, n_jobs: int):
    """
    The model. If the candidates run in parallel (n_jobs > 1) it gets one thread: the models use
    all the cores by default, and with the two levels the threads pile up and everything slows down.
    """
    model = build_model(model_name, params=params)
    if n_jobs > 1 and "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    return model


def _pick_best(model_name: str, cv_results: dict) -> int:
    """Index of the simplest candidate among those within the tolerance of the best mean."""
    means = np.asarray(cv_results["mean_test_score"])
    tied = np.flatnonzero(means >= means.max() - TUNING_TIE_TOLERANCE)
    return min(tied, key=lambda i: COMPLEXITY[model_name](cv_results["params"][i]))
