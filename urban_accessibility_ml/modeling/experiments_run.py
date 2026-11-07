"""Run the experiment matrix with the default hyperparameters and with the tuned ones, and write the results.

make experiments
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from urban_accessibility_ml.config import REPORTS_DIR
from urban_accessibility_ml.data.pipeline import UNIFIED_FILE
from urban_accessibility_ml.modeling.experiments import run_experiments, save_results
from urban_accessibility_ml.modeling.tuning.tuning_results import load_best_params
from urban_accessibility_ml.preprocessing.split import build_all_splits

app = typer.Typer(help=__doc__)

BEST_PARAMS_FILE = REPORTS_DIR / "results" / "best_params.json"
DEFAULT_FILE = REPORTS_DIR / "results" / "experiment-matrix-default.csv"
TUNED_FILE = REPORTS_DIR / "results" / "experiment-matrix-tuned.csv"


def run_matrix(
    df: pd.DataFrame,
    best_params_path: Path = BEST_PARAMS_FILE,
    default_path: Path = DEFAULT_FILE,
    tuned_path: Path = TUNED_FILE,
    overwrite: bool = False,
) -> None:
    """
    Run the matrix with the default hyperparameters and with the tuned ones, and write one CSV for each.

    :param df: unified dataset.
    :param best_params_path: JSON of the tuned hyperparameters (make tune).
    :param overwrite: write over the CSV files if they exist.
    :raises FileExistsError: if a CSV exists and overwrite is False.
    :raises FileNotFoundError: if best_params_path does not exist.
    """
    existing = [p for p in (default_path, tuned_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Already exist (use --overwrite): {[str(p) for p in existing]}")
    best_params = load_best_params(best_params_path)

    splits = build_all_splits(df)
    save_results(run_experiments(df, splits), default_path)
    save_results(run_experiments(df, splits, hyperparameters=best_params), tuned_path)
    logger.info(f"Saved {default_path} and {tuned_path}")


@app.command()
def main(
    overwrite: bool = typer.Option(False, help="Write over the result files if they exist."),
) -> None:
    """Run the experiment matrix with the default and the tuned hyperparameters (takes several minutes)."""
    try:
        run_matrix(pd.read_csv(UNIFIED_FILE), overwrite=overwrite)
    except (FileExistsError, FileNotFoundError) as e:
        logger.error(e)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
