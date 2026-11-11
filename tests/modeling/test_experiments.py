import numpy as np
import pandas as pd
import pytest

from urban_accessibility_ml.modeling import experiments
from urban_accessibility_ml.modeling.experiments import run_experiments, save_results
from urban_accessibility_ml.modeling.tuning.tuning_results import tuning_key
from urban_accessibility_ml.preprocessing.split import Split, build_all_splits

FEATURE_SET = "completo"
PROFILE = "global"
MODEL_NAME = "dummy"
SPLIT = Split("within_city", "Albacete", 2, np.array(["t1"]), np.array(["t2"]))
RESULTS_FILE = "results.csv"
UNKNOWN_FEATURE_SET = "does_not_exist"
UNKNOWN_PROFILE = "does_not_exist"
UNKNOWN_MODEL = "does_not_exist"


def test_run_experiments_one_element_lists(city_df):
    splits = build_all_splits(city_df)
    results = run_experiments(city_df, splits, [FEATURE_SET], [PROFILE], [MODEL_NAME])
    assert len(results) == len(splits)
    assert set(results["model"]) == {MODEL_NAME}


def test_run_experiments_rows_are_the_product_of_the_lists(city_df):
    splits = build_all_splits(city_df)[:3]
    feature_sets, profiles, models = ["calle", "completo"], ["global", "funcional"], ["dummy", "logistic_regression"]
    results = run_experiments(city_df, splits, feature_sets, profiles, models)
    assert len(results) == len(splits) * len(feature_sets) * len(profiles) * len(models)
    assert set(results["feature_set"]) == set(feature_sets)
    assert set(results["profile"]) == set(profiles)


def test_run_experiments_defaults_come_from_config(city_df, monkeypatch):
    monkeypatch.setattr(experiments, "FEATURE_SETS", ["calle"])
    monkeypatch.setattr(experiments, "PROFILES", ["global", "funcional"])
    monkeypatch.setattr(experiments, "MATRIX_MODELS", ["dummy"])
    splits = build_all_splits(city_df)[:2]
    results = run_experiments(city_df, splits)
    assert len(results) == 2 * 1 * 2 * 1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"feature_sets": [UNKNOWN_FEATURE_SET]}, "feature set"),
        ({"profiles": [UNKNOWN_PROFILE]}, "profile"),
        ({"model_names": [UNKNOWN_MODEL]}, "model"),
    ],
)
def test_run_experiments_unknown_name_fails_before_training(city_df, monkeypatch, kwargs, match):
    monkeypatch.setattr(experiments, "run_one", lambda *a: pytest.fail("trained before validating"))
    with pytest.raises(ValueError, match=match):
        run_experiments(city_df, build_all_splits(city_df), **kwargs)


def test_run_experiments_does_not_modify_df(city_df):
    plain = city_df.reset_index(drop=True)  # key_tramo only as a column
    before = plain.copy()
    run_experiments(plain, build_all_splits(plain)[:1], [FEATURE_SET], [PROFILE], [MODEL_NAME])
    pd.testing.assert_frame_equal(plain, before)


def test_run_experiments_without_key_tramo_fails(city_df):
    with pytest.raises(ValueError, match="key_tramo"):
        run_experiments(city_df.drop(columns="key_tramo").reset_index(drop=True), [SPLIT], [FEATURE_SET], [PROFILE], [MODEL_NAME])


def test_save_results_creates_folders_and_round_trips(city_df, tmp_path):
    results = run_experiments(city_df, build_all_splits(city_df)[:3], [FEATURE_SET], [PROFILE], [MODEL_NAME])
    path = tmp_path / "a" / "b" / RESULTS_FILE
    assert save_results(results, path) == path
    assert path.exists()
    read = pd.read_csv(path)
    assert "Unnamed: 0" not in read.columns
    pd.testing.assert_frame_equal(read, results)


def test_run_experiments_passes_hyperparameters_to_every_run(city_df, monkeypatch):
    received = []
    monkeypatch.setattr(experiments, "run_one", lambda *a: received.append(a[5]) or {"model": a[4]})
    hyperparameters = {"some/partition/0": {"logistic_regression": {"C": 0.1}}}
    splits = build_all_splits(city_df)[:2]
    run_experiments(city_df, splits, [FEATURE_SET], [PROFILE], ["dummy"], hyperparameters)
    assert received == [hyperparameters, hyperparameters]


def test_run_experiments_without_hyperparameters_gives_none_to_run_one(city_df, monkeypatch):
    received = []
    monkeypatch.setattr(experiments, "run_one", lambda *a: received.append(a[5]) or {"model": a[4]})
    run_experiments(city_df, build_all_splits(city_df)[:1], [FEATURE_SET], [PROFILE], ["dummy"])
    assert received == [None]


def test_run_experiments_with_tuned_params_end_to_end(city_df):
    splits = build_all_splits(city_df)[:2]
    hyperparameters = {tuning_key(s, FEATURE_SET, "logistic_regression"): {"C": 0.1} for s in splits}
    results = run_experiments(
        city_df, splits, [FEATURE_SET], [PROFILE], ["dummy", "logistic_regression"], hyperparameters
    )
    assert len(results) == len(splits) * 2



def test_run_experiments_passes_threshold_to_every_run(city_df, monkeypatch):
    received = []
    monkeypatch.setattr(experiments, "run_one", lambda *a: received.append(a[6]) or {"model": a[4]})
    run_experiments(
        city_df, build_all_splits(city_df)[:1], [FEATURE_SET], [PROFILE], ["dummy"], threshold=75
    )
    assert received == [75]


def test_run_experiments_passes_fallback_to_every_run(city_df, monkeypatch):
    received = []
    monkeypatch.setattr(experiments, "run_one", lambda *a: received.append(a[7]) or {"model": a[4]})
    splits = build_all_splits(city_df)[:1]
    run_experiments(city_df, splits, [FEATURE_SET], [PROFILE], ["dummy"], fallback_to_complete=True)
    run_experiments(city_df, splits, [FEATURE_SET], [PROFILE], ["dummy"])
    assert received == [True, False]
