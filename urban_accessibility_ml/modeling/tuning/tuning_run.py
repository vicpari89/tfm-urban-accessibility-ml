"""Run the hyperparameter tuning in the 17 partitions and write the results to reports/results/.

make tune
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from urban_accessibility_ml.config import REPORTS_DIR
from urban_accessibility_ml.data.pipeline import UNIFIED_FILE
from urban_accessibility_ml.modeling.tuning.tuning_results import save_best_params, save_candidates
from urban_accessibility_ml.modeling.tuning.tuning_search import tune_all
from urban_accessibility_ml.preprocessing.split import build_all_splits

app = typer.Typer(help=__doc__)

BEST_PARAMS_FILE = REPORTS_DIR / "results" / "best_params.json"
CANDIDATES_FILE = REPORTS_DIR / "results" / "tuning-candidates.csv"


def run_tuning(
    df: pd.DataFrame,
    best_params_path: Path = BEST_PARAMS_FILE,
    candidates_path: Path = CANDIDATES_FILE,
    overwrite: bool = False,
) -> None:
    """
    Tune every model in every partition of build_all_splits and write the two result files.

    :param df: unified dataset (data/processed/streets_unified.csv).
    :param best_params_path: JSON with the best hyperparameters of each partition and model.
    :param candidates_path: CSV with all the candidates evaluated.
    :param overwrite: write over the files if they already exist.
    :raises FileExistsError: if a file exists and overwrite is False. It is checked before tuning,
        because the tuning takes a long time.
    """
    existing = [p for p in (best_params_path, candidates_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Already exist (use --overwrite): {[str(p) for p in existing]}")

    best_params, candidates = tune_all(df, build_all_splits(df))
    save_best_params(best_params, best_params_path)
    save_candidates(candidates, candidates_path)
    logger.info(f"Saved {best_params_path} and {candidates_path}")


@app.command()
def main(
    overwrite: bool = typer.Option(False, help="Write over the result files if they exist."),
) -> None:
    """Tune the hyperparameters of the models in the 17 partitions (takes about 30 minutes)."""
    try:
        run_tuning(pd.read_csv(UNIFIED_FILE), overwrite=overwrite)
    except (FileExistsError, FileNotFoundError) as e:
        logger.error(e)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
