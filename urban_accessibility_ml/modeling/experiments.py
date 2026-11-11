"""Experiment matrix: run_one (train.py) for every combination, and save the results.

preprocessing/split.build_all_splits -> run_experiments -> save_results
"""

from itertools import product
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from urban_accessibility_ml.config import FEATURE_SETS, MATRIX_MODELS, PROFILES
from urban_accessibility_ml.modeling.models import BUILDERS
from urban_accessibility_ml.modeling.train import run_one
from urban_accessibility_ml.preprocessing.split import ID_COL, Split


def run_experiments(
    df: pd.DataFrame,
    splits: list[Split],
    feature_sets: list[str] | None = None,
    profiles: list[str] | None = None,
    model_names: list[str] | None = None,
    hyperparameters: dict | None = None,
    threshold: int | None = None,
    fallback_to_complete: bool = False,
) -> pd.DataFrame:
    """
    Run every combination of split x feature set x profile x model with run_one.

    :param df: unified dataset, with key_tramo included.
    :param splits: partitions to run (build_all_splits).
    :param feature_sets: feature sets to combine. By default config.FEATURE_SETS.
    :param profiles: profiles to combine. By default config.PROFILES.
    :param model_names: models to combine. By default config.MATRIX_MODELS.
    :param hyperparameters: {tuning_key: params} (best_params.json). None: the model defaults.
    :param threshold: viability threshold of the feature sets. None: config.THRESHOLD.
    :param fallback_to_complete: use the "completo" hyperparameters when a combination has none.
    :return: one row per model and fold.
    :raises ValueError: if a feature set, a profile or a model does not exist.
    """
    feature_sets = FEATURE_SETS if feature_sets is None else feature_sets
    profiles = PROFILES if profiles is None else profiles
    model_names = MATRIX_MODELS if model_names is None else model_names

    # 1. Validate the names
    _check_names("feature set", feature_sets, FEATURE_SETS)
    _check_names("profile", profiles, PROFILES)
    _check_names("model", model_names, list(BUILDERS))

    # 2. run_one needs key_tramo as index
    indexed = _index_by_key(df)

    # 3. One run_one per combination, with a progress bar
    combinations = list(product(splits, feature_sets, profiles, model_names))
    rows = [
        run_one(
            indexed,
            split,
            feature_set,
            profile,
            model_name,
            hyperparameters,
            threshold,
            fallback_to_complete,
        )
        for split, feature_set, profile, model_name in tqdm(combinations, desc="experiments")
    ]
    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame, path: str | Path) -> Path:
    """
    Write the results to a CSV, without the index column.

    :param results: the DataFrame of run_experiments.
    :param path: CSV file to write.
    :return: the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path


def _check_names(kind: str, names: list[str], valid: list[str]) -> None:
    """Raise ValueError if some of the names is not in valid."""
    unknown = [n for n in names if n not in valid]
    if unknown:
        raise ValueError(f"Unknown {kind}: {unknown}. Options: {valid}")


def _index_by_key(df: pd.DataFrame) -> pd.DataFrame:
    """The dataset indexed by key_tramo, without modifying the original."""
    if df.index.name == ID_COL:
        return df
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} must be the index or a column of the dataframe")
    return df.set_index(ID_COL, drop=False)
