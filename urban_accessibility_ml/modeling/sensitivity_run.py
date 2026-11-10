"""Sensitivity analysis of the viability threshold: run the matrix for every distinct feature set.

make sensitivity
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
import pandas as pd
import typer

from urban_accessibility_ml.config import REPORTS_DIR
from urban_accessibility_ml.data.pipeline import UNIFIED_FILE
from urban_accessibility_ml.modeling.experiments import run_experiments, save_results
from urban_accessibility_ml.modeling.tuning.tuning_constants import SEARCH_SPACES
from urban_accessibility_ml.modeling.tuning.tuning_results import load_best_params, tuning_key
from urban_accessibility_ml.preprocessing.features import sensitivity_sets
from urban_accessibility_ml.preprocessing.split import build_all_splits

app = typer.Typer(help=__doc__)

BEST_PARAMS_FILE = REPORTS_DIR / "results" / "best_params.json"
SENSITIVITY_FILE = REPORTS_DIR / "results" / "sensitivity.csv"


def run_sensitivity(
    df: pd.DataFrame,
    best_params: dict,
    splits: list | None = None,
    profiles: list[str] | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run the matrix once per distinct set of dimensions (sensitivity_sets) and copy the results to
    every threshold and feature set that has them. Hyperparameters fall back to the "completo" ones.

    :param df: unified dataset.
    :param best_params: {tuning_key: params} (best_params.json).
    :param splits, profiles, model_names: None runs the whole matrix.
    :return: results with threshold, feature_set and dimensions first, used_fallback last.
    """
    table, unique = sensitivity_sets()
    splits = build_all_splits(df) if splits is None else splits
    parts = []
    for dims in unique:
        rows = table[[d == dims for d in table["dimensions"]]]
        results = _run_set(df, splits, rows.iloc[0], best_params, profiles, model_names)
        parts += [_label(results, row, dims) for row in rows.itertuples()]
    return _label_columns_first(pd.concat(parts, ignore_index=True))


def _run_set(df, splits, first, best_params, profiles, model_names) -> pd.DataFrame:
    """The matrix for one set (a row of sensitivity_sets), with used_fallback."""
    results = run_experiments(
        df,
        splits,
        [first.feature_set],
        profiles,
        model_names,
        best_params,
        threshold=first.threshold,
        fallback_to_complete=True,
    )
    by_key = {s.key: s for s in splits}
    results["used_fallback"] = [
        _used_fallback(r, by_key, first, best_params) for r in results.itertuples()
    ]
    return results


def _used_fallback(result, by_key: dict, first, best_params: dict) -> bool:
    """True if a tuned model had no entry of its own and used the one of "completo"."""
    if result.model not in SEARCH_SPACES:
        return False
    split = by_key[f"{result.strategy}/{result.name}/{result.fold}"]
    return tuning_key(split, first.feature_set, result.model, first.threshold) not in best_params


def _label(results: pd.DataFrame, row, dims: tuple[str, ...]) -> pd.DataFrame:
    """The results as those of a threshold and feature set."""
    return results.assign(
        threshold=row.threshold, feature_set=row.feature_set, dimensions=",".join(dims)
    )


def _label_columns_first(out: pd.DataFrame) -> pd.DataFrame:
    """threshold, feature_set and dimensions as the first columns."""
    first_cols = ["threshold", "feature_set", "dimensions"]
    return out[first_cols + [c for c in out.columns if c not in first_cols]]


@app.command()
def main(
    best_params_path: Annotated[
        Path, typer.Option(help="JSON of the tuned params.")
    ] = BEST_PARAMS_FILE,
    output: Annotated[Path, typer.Option(help="CSV to write.")] = SENSITIVITY_FILE,
    overwrite: Annotated[
        bool, typer.Option(help="Write over sensitivity.csv if it exists.")
    ] = False,
) -> None:
    """Run the matrix for every distinct feature set of every threshold (takes several minutes)."""
    if output.exists() and not overwrite:
        logger.error(f"Already exists (use --overwrite): {output}")
        raise typer.Exit(code=1)
    try:
        best_params = load_best_params(best_params_path)
    except FileNotFoundError as e:
        logger.error(e)
        raise typer.Exit(code=1)
    save_results(run_sensitivity(pd.read_csv(UNIFIED_FILE), best_params), output)
    logger.info(f"Saved {output}")


if __name__ == "__main__":
    app()
