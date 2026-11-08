"""Write and read the results of the tuning (tuning_search.py).

tuning_search.tune_all -> save_best_params (JSON) and save_candidates (CSV)
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
    Write the best params of tune_all ({Split.key: {model: params}}) as a JSON.

    :param best_params: the first result of tune_all.
    :param path: JSON file to write. Its folders are created if they do not exist.
    :return: the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(best_params, indent=2, default=_to_python) + "\n")
    return path


def load_best_params(path: str | Path) -> dict:
    """
    Read the JSON of save_best_params.

    :param path: JSON file to read.
    :return: {Split.key: {model: params}}, ready for run_one and run_experiments.
    """
    return json.loads(Path(path).read_text())


def save_candidates(candidates: pd.DataFrame, path: str | Path) -> Path:
    """
    Write all the candidates evaluated to a CSV, without the index column. The params of each
    candidate are written as a JSON text.

    :param candidates: the second result of tune_all.
    :param path: CSV file to write. Its folders are created if they do not exist.
    :return: the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    as_text = candidates["params"].map(lambda params: json.dumps(params, default=_to_python))
    candidates.assign(params=as_text).to_csv(path, index=False)
    return path


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
