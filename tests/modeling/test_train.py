import numpy as np
import pandas as pd
import pytest

from urban_accessibility_ml.config import DIMENSIONS, METRICS, PROFILES, TARGET_COLUMNS
from urban_accessibility_ml.modeling.train import run_one
from urban_accessibility_ml.preprocessing.split import Split

N_ROWS = 60
FEATURE_SET = "completo"
PROFILE = "global"
MODEL_NAME = "dummy"
TRAIN_IDS = np.array([f"t{i}" for i in range(40)])
TEST_IDS = np.array([f"t{i}" for i in range(40, N_ROWS)])
SPLIT = Split("within_city", "Albacete", 2, TRAIN_IDS, TEST_IDS)
UNKNOWN_MODEL = "does_not_exist"


@pytest.fixture
def df():
    classes = ["C", "B", "A"] * (N_ROWS // 3)
    data = {f"puntuacion_{d}_{p}": np.linspace(0, 1, N_ROWS) for d in DIMENSIONS for p in PROFILES}
    data |= {col: classes for col in TARGET_COLUMNS.values()}
    return pd.DataFrame(data, index=pd.Index([f"t{i}" for i in range(N_ROWS)], name="key_tramo"))


def test_run_one_row_fields(df):
    row = run_one(df, SPLIT, FEATURE_SET, PROFILE, MODEL_NAME)
    assert row["strategy"] == "within_city"
    assert row["name"] == "Albacete"
    assert row["fold"] == 2
    assert row["feature_set"] == FEATURE_SET
    assert row["profile"] == PROFILE
    assert row["model"] == MODEL_NAME


def test_run_one_set_sizes(df):
    row = run_one(df, SPLIT, FEATURE_SET, PROFILE, MODEL_NAME)
    assert row["n_train"] == len(TRAIN_IDS)
    assert row["n_test"] == len(TEST_IDS)


def test_run_one_has_every_metric(df):
    row = run_one(df, SPLIT, FEATURE_SET, PROFILE, MODEL_NAME)
    assert all(0 <= row[m] <= 1 for m in METRICS if m != "quadratic_weighted_kappa")
    assert "quadratic_weighted_kappa" in row


def test_run_one_dummy_scores_one_third_of_balanced_accuracy(df):
    row = run_one(df, SPLIT, FEATURE_SET, PROFILE, MODEL_NAME)
    assert row["balanced_accuracy"] == pytest.approx(1 / 3)


def test_run_one_unknown_model(df):
    with pytest.raises(ValueError, match="Unknown model"):
        run_one(df, SPLIT, FEATURE_SET, PROFILE, UNKNOWN_MODEL)


def test_run_one_unknown_feature_set(df):
    with pytest.raises(ValueError):
        run_one(df, SPLIT, "does_not_exist", PROFILE, MODEL_NAME)


def test_run_one_rows_collect_into_a_dataframe(df):
    rows = [run_one(df, SPLIT, FEATURE_SET, PROFILE, m) for m in ["dummy", "logistic_regression"]]
    assert pd.DataFrame(rows)["model"].tolist() == ["dummy", "logistic_regression"]
