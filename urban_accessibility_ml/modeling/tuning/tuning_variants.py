"""Tuning of the tree models with another seed and with more candidates, to see how much the
results depend on chance. The main tuning (make tune) is the reference.

make tune-variants
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
import pandas as pd
import typer

from urban_accessibility_ml.config import REPORTS_DIR
from urban_accessibility_ml.data.pipeline import UNIFIED_FILE
from urban_accessibility_ml.modeling.experiments import run_experiments, save_results
from urban_accessibility_ml.modeling.tuning.tuning_constants import (
    SEARCH_SPACES,
    TUNING_N_CANDIDATES,
)
from urban_accessibility_ml.modeling.tuning.tuning_results import load_best_params
from urban_accessibility_ml.modeling.tuning.tuning_run import (
    BEST_PARAMS_FILE,
    DEFAULT_N_JOBS,
    run_tuning,
)
from urban_accessibility_ml.preprocessing.split import build_all_splits

app = typer.Typer(help=__doc__)

VARIANTS_DIR = REPORTS_DIR / "results" / "variants"
# Each variant changes one thing of the main tuning. A seed of None is config.SEED.
VARIANTS = {
    "seed2": {"seed": 123, "n_candidates": TUNING_N_CANDIDATES},
    "n100": {"seed": None, "n_candidates": 100},
}
# The logistic regression is a grid (always the same 16 candidates): it keeps the main tuning.
TREE_MODELS = [m for m, spec in SEARCH_SPACES.items() if spec["kind"] == "random"]


def run_variants(
    df: pd.DataFrame,
    names: list[str] | None = None,
    best_params_path: Path = BEST_PARAMS_FILE,
    out_dir: Path = VARIANTS_DIR,
    overwrite: bool = False,
    n_jobs: int = DEFAULT_N_JOBS,
    dry_run: bool = False,
) -> None:
    """
    For each variant, tune the tree models and run the experiment matrix with those values (and
    the logistic regression of the main tuning). It continues where a stopped run was.

    :param df: unified dataset.
    :param names: keys of VARIANTS. By default all.
    :param best_params_path: JSON of the main tuning (make tune).
    :param out_dir: folder of the results, with the name of each variant in its files.
    :param overwrite: start from scratch.
    :param dry_run: only estimate the time of the tuning.
    :raises ValueError: if a variant does not exist.
    :raises FileNotFoundError: if best_params_path does not exist.
    """
    names = list(VARIANTS) if names is None else names
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant: {unknown}. Options: {list(VARIANTS)}")
    main = load_best_params(best_params_path)  # fails here, before the long tuning

    splits = build_all_splits(df)
    for name in names:
        logger.info(f"Variant {name}: {VARIANTS[name]}")
        best = out_dir / f"best_params_{name}.json"
        candidates = out_dir / f"tuning-candidates_{name}.csv"
        matrix = out_dir / f"experiment-matrix_{name}.csv"
        run_tuning(
            df, best, candidates, overwrite, n_jobs, dry_run, models=TREE_MODELS, **VARIANTS[name]
        )
        if dry_run:
            continue
        if matrix.exists() and not overwrite:
            logger.info(f"{matrix} already exists, skipped")
            continue
        others = {k: v for k, v in main.items() if k.split("|")[2] not in TREE_MODELS}
        hyperparameters = {**others, **load_best_params(best)}
        save_results(run_experiments(df, splits, hyperparameters=hyperparameters), matrix)
        logger.info(f"Saved {matrix}")


@app.command()
def main(
    variant: Annotated[
        list[str] | None, typer.Option(help="Variant to run (repeat it). By default all.")
    ] = None,
    overwrite: bool = typer.Option(False, help="Start from scratch."),
    n_jobs: int = typer.Option(DEFAULT_N_JOBS, help="Candidates fitted in parallel."),
    dry_run: bool = typer.Option(False, help="Only estimate the time of the tuning."),
) -> None:
    """Tune the tree models with another seed and with more candidates (takes about an hour)."""
    try:
        run_variants(
            pd.read_csv(UNIFIED_FILE),
            variant or None,
            overwrite=overwrite,
            n_jobs=n_jobs,
            dry_run=dry_run,
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error(e)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
