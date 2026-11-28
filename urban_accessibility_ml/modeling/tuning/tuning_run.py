"""Run the hyperparameter tuning in the 17 partitions and the feature sets, and write the results
to reports/results/.

make tune
"""

import os
from pathlib import Path
import warnings

from loguru import logger
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm
import typer

from urban_accessibility_ml.config import FEATURE_SETS, REPORTS_DIR
from urban_accessibility_ml.data.pipeline import UNIFIED_FILE
from urban_accessibility_ml.modeling.experiments import _index_by_key
from urban_accessibility_ml.modeling.tuning.tuning_constants import (
    SEARCH_SPACES,
    TUNING_FEATURE_SET,
    TUNING_FEATURE_SET_ORDER,
)
from urban_accessibility_ml.modeling.tuning.tuning_results import (
    is_tuned,
    load_best_params,
    save_tuning,
    tuning_key,
)
from urban_accessibility_ml.modeling.tuning.tuning_search import (
    time_candidate,
    train_data,
    tune_partition,
)
from urban_accessibility_ml.preprocessing.features import dimensions_for, sensitivity_sets
from urban_accessibility_ml.preprocessing.split import build_all_splits

app = typer.Typer(help=__doc__)

BEST_PARAMS_FILE = REPORTS_DIR / "results" / "best_params.json"
CANDIDATES_FILE = REPORTS_DIR / "results" / "tuning-candidates.csv"
DEFAULT_N_JOBS = max(1, (os.cpu_count() or 2) // 2)  # half of the cores, for 16 GB of RAM


def sensitivity_units() -> list[tuple[str, int]]:
    """
    (feature set, threshold) of each set of dimensions of sensitivity_sets that is not tuned yet,
    that is, the ones that are not those of the main threshold or "completo".
    """
    tuned = {tuple(dimensions_for(fs)) for fs in FEATURE_SETS}
    table, distinct = sensitivity_sets()
    units = []
    for dims in distinct:
        if dims not in tuned:
            first = table[[d == dims for d in table["dimensions"]]].iloc[0]
            units.append((first.feature_set, int(first.threshold)))
    return units


def tuning_units(sensitivity: bool = False) -> list[tuple[str, int | None]]:
    """
    (feature set, threshold) to tune, in order: the ones of the main threshold (threshold None) and,
    if sensitivity, the extra sets of the sensitivity analysis.
    """
    units = [(fs, None) for fs in TUNING_FEATURE_SET_ORDER]
    return units + sensitivity_units() if sensitivity else units


def estimate_seconds(
    indexed: pd.DataFrame,
    splits: list,
    saved: dict,
    n_jobs: int,
    models: list[str],
    seed: int | None = None,
    n_candidates: int | None = None,
    units: list[tuple[str, int | None]] | None = None,
) -> float:
    """
    Estimated seconds left, from timing the models on the biggest partition and dividing by n_jobs.

    :param units: (feature set, threshold) to tune. None: the ones of the main threshold.
    """
    pending = {  # calle and combinado are the same tuning
        tuning_key(s, fs, "", t)
        for fs, t in (tuning_units() if units is None else units)
        for s in splits
        if not is_tuned(saved, s, fs, models, t)
    }
    if not pending:
        return 0.0
    biggest = max(splits, key=lambda s: len(s.train_ids))
    data = train_data(indexed, biggest, TUNING_FEATURE_SET)
    per_unit = 0.0
    for model_name in models:
        seconds, total = time_candidate(model_name, *data, n_jobs, seed, n_candidates)
        logger.info(f"{model_name}: {seconds:.1f} s per candidate, {total} candidates")
        per_unit += seconds * total
    return per_unit * len(pending) / n_jobs


def run_tuning(
    df: pd.DataFrame,
    best_params_path: Path = BEST_PARAMS_FILE,
    candidates_path: Path = CANDIDATES_FILE,
    overwrite: bool = False,
    n_jobs: int = DEFAULT_N_JOBS,
    dry_run: bool = False,
    seed: int | None = None,
    n_candidates: int | None = None,
    models: list[str] | None = None,
    sensitivity: bool = False,
) -> None:
    """
    Tune every model in every partition and feature set, in TUNING_FEATURE_SET_ORDER. Each result
    is saved when it ends and the ones already saved are skipped, so a stopped run continues.

    :param df: unified dataset.
    :param overwrite: delete the two files first and start from scratch.
    :param n_jobs: number of parallel runs.
    :param seed: seed of the inner folds and the random search. By default config.SEED.
    :param n_candidates: candidates of a random search. By default TUNING_N_CANDIDATES.
    :param models: models to tune. By default all of SEARCH_SPACES.
    :param sensitivity: also tune the sets of dimensions of the sensitivity analysis (after the
        others), with the same procedure, seed and candidates.
    """
    # saga does not converge in some candidates; it is not an error (sklearn sends it to the workers)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if overwrite:
        for path in (best_params_path, candidates_path):
            path.unlink(missing_ok=True)

    models = list(SEARCH_SPACES) if models is None else models
    indexed, splits = _index_by_key(df), build_all_splits(df)
    saved = load_best_params(best_params_path) if best_params_path.exists() else {}
    units = tuning_units(sensitivity)
    seconds = estimate_seconds(indexed, splits, saved, n_jobs, models, seed, n_candidates, units)
    logger.info(f"Estimated time left: {seconds / 60:.0f} min")
    if dry_run:
        return
    for feature_set, threshold in units:
        for split in tqdm(
            splits, desc=feature_set if threshold is None else f"{feature_set}@{threshold}"
        ):
            saved = load_best_params(best_params_path) if best_params_path.exists() else {}
            if is_tuned(saved, split, feature_set, models, threshold):
                continue
            best_params, candidates = tune_partition(
                indexed, split, feature_set, n_jobs, seed, n_candidates, models, threshold
            )
            save_tuning(
                split,
                feature_set,
                best_params,
                candidates,
                best_params_path,
                candidates_path,
                threshold,
            )
    logger.info(f"Saved {best_params_path} and {candidates_path}")


@app.command()
def main(
    overwrite: bool = typer.Option(False, help="Delete the result files and start from scratch."),
    n_jobs: int = typer.Option(DEFAULT_N_JOBS, help="Candidates fitted in parallel."),
    dry_run: bool = typer.Option(False, help="Only estimate the time left."),
) -> None:
    """Tune the models in the 17 partitions and the feature sets, also those of the sensitivity
    analysis. Continues a stopped run."""
    try:
        run_tuning(
            pd.read_csv(UNIFIED_FILE),
            overwrite=overwrite,
            n_jobs=n_jobs,
            dry_run=dry_run,
            sensitivity=True,
        )
    except FileNotFoundError as e:
        logger.error(e)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
