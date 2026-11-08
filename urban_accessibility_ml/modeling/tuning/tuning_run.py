"""Run the hyperparameter tuning in the 17 partitions and the feature sets, and write the results
to reports/results/.

make tune
"""

import os
from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm.auto import tqdm
import typer

from urban_accessibility_ml.config import REPORTS_DIR
from urban_accessibility_ml.data.pipeline import UNIFIED_FILE
from urban_accessibility_ml.modeling.experiments import _index_by_key
from urban_accessibility_ml.modeling.tuning.tuning_constants import (
    SEARCH_SPACES,
    TUNING_FEATURE_SET_ORDER,
)
from urban_accessibility_ml.modeling.tuning.tuning_results import (
    is_tuned,
    load_best_params,
    save_tuning,
)
from urban_accessibility_ml.modeling.tuning.tuning_search import tune_partition
from urban_accessibility_ml.preprocessing.split import build_all_splits

app = typer.Typer(help=__doc__)

BEST_PARAMS_FILE = REPORTS_DIR / "results" / "best_params.json"
CANDIDATES_FILE = REPORTS_DIR / "results" / "tuning-candidates.csv"
DEFAULT_N_JOBS = max(1, (os.cpu_count() or 2) // 2)  # half of the cores, for 16 GB of RAM


def run_tuning(
    df: pd.DataFrame,
    best_params_path: Path = BEST_PARAMS_FILE,
    candidates_path: Path = CANDIDATES_FILE,
    overwrite: bool = False,
    n_jobs: int = DEFAULT_N_JOBS,
) -> None:
    """
    Tune every model in every partition and feature set, in TUNING_FEATURE_SET_ORDER. Each result
    is saved when it ends and the ones already saved are skipped, so a stopped run continues.

    :param df: unified dataset.
    :param overwrite: delete the two files first and start from scratch.
    :param n_jobs: number of parallel runs.
    """
    if overwrite:
        for path in (best_params_path, candidates_path):
            path.unlink(missing_ok=True)

    indexed, splits = _index_by_key(df), build_all_splits(df)
    for feature_set in TUNING_FEATURE_SET_ORDER:
        for split in tqdm(splits, desc=feature_set):
            saved = load_best_params(best_params_path) if best_params_path.exists() else {}
            if is_tuned(saved, split, feature_set, SEARCH_SPACES):
                continue
            best_params, candidates = tune_partition(indexed, split, feature_set, n_jobs)
            save_tuning(
                split, feature_set, best_params, candidates, best_params_path, candidates_path
            )
    logger.info(f"Saved {best_params_path} and {candidates_path}")


@app.command()
def main(
    overwrite: bool = typer.Option(False, help="Delete the result files and start from scratch."),
    n_jobs: int = typer.Option(DEFAULT_N_JOBS, help="Candidates fitted in parallel."),
) -> None:
    """Tune the models in the 17 partitions and the feature sets. Continues a stopped run."""
    try:
        run_tuning(pd.read_csv(UNIFIED_FILE), overwrite=overwrite, n_jobs=n_jobs)
    except FileNotFoundError as e:
        logger.error(e)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
