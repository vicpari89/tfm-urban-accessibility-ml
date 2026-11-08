"""Write and read the results of the tuning (tuning_search.py).

tuning_run.run_tuning -> save_tuning -> best_params (JSON) and candidates (CSV)
load_best_params (JSON) -> train.run_one and experiments.run_experiments
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from urban_accessibility_ml.preprocessing.features import dimensions_for
from urban_accessibility_ml.preprocessing.split import Split


def save_best_params(best_params: dict, path: str | Path) -> Path:
    """
    Write the best params ({tuning_key: params}) as a JSON. The file is replaced at once,
        to prevent interruption problems.

    :param best_params: best params by tuning_key.
    :param path: JSON file to write. Its folders are created if they do not exist.
    :return: the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, json.dumps(best_params, indent=2, default=_to_python) + "\n")
    return path


def load_best_params(path: str | Path) -> dict:
    """
    Read the JSON of save_best_params.

    :param path: JSON file to read.
    :return: {tuning_key: params}.
    """
    return json.loads(Path(path).read_text())


def save_candidates(candidates: pd.DataFrame, path: str | Path) -> Path:
    """
    Write all the candidates evaluated to a CSV, without the index column. The params of each
    candidate are written as a JSON text.

    :param candidates: the candidates of the tuning.
    :param path: CSV file to write. Its folders are created if they do not exist.
    :return: the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, _params_as_text(candidates).to_csv(index=False))
    return path


def is_tuned(best_params: dict, split: Split, feature_set: str, models) -> bool:
    """True if best_params already has every model for this partition and feature set."""
    return all(tuning_key(split, feature_set, m) in best_params for m in models)


def save_tuning(
    split: Split,
    feature_set: str,
    best_params: dict,
    candidates: pd.DataFrame,
    best_params_path: str | Path,
    candidates_path: str | Path,
) -> None:
    """
    Add the result of one (partition, feature set) to the two files. The best params are written
    last: a tuning counts as done (is_tuned) only when they are there.
    """
    best_params_path, candidates_path = Path(best_params_path), Path(candidates_path)
    unit = ["strategy", "name", "fold", "feature_set"]
    new = _params_as_text(candidates.assign(feature_set=feature_set))
    if candidates_path.exists():
        old = pd.read_csv(candidates_path)
        new = pd.concat([old[~(old[unit] == new[unit].iloc[0]).all(axis=1)], new])
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(candidates_path, new.to_csv(index=False))

    saved = load_best_params(best_params_path) if best_params_path.exists() else {}
    saved.update({tuning_key(split, feature_set, m): p for m, p in best_params.items()})
    save_best_params(saved, best_params_path)


def _params_as_text(candidates: pd.DataFrame) -> pd.DataFrame:
    """The candidates with the params column as a JSON text."""
    as_text = candidates["params"].map(lambda params: json.dumps(params, default=_to_python))
    return candidates.assign(params=as_text)


def _write_atomic(path: Path, text: str) -> None:
    """Write to a temporary file and rename it, so the file is never half written."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _to_python(value):
    """
    For json.dumps: a numpy number (np.int64, np.float64, ...) as a python one. The random
    searches of tuning_search.py sample numpy numbers, which json does not know how to write.
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def tuning_key(split: Split, feature_set: str, model: str) -> str:
    """
    Key of a tuning in best_params.json, e.g. "within_city/Albacete/0|cruces,mobiliario|lightgbm".
    It uses the dimensions of the feature set, not its name: sets with the same dimensions share
    the tuning.
    """
    dimensions = ",".join(sorted(dimensions_for(feature_set)))
    return f"{split.key}|{dimensions}|{model}"
