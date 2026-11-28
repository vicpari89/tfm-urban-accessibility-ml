import pandas as pd
import pytest
from typer.testing import CliRunner

from urban_accessibility_ml.config import THRESHOLDS
from urban_accessibility_ml.modeling import sensitivity_run
from urban_accessibility_ml.modeling.sensitivity_run import app, run_sensitivity
from urban_accessibility_ml.modeling.tuning.tuning_results import tuning_key
from urban_accessibility_ml.preprocessing.features import sensitivity_sets
from urban_accessibility_ml.preprocessing.split import build_all_splits


@pytest.fixture
def splits(city_df):
    return build_all_splits(city_df)[:2]


def test_one_row_per_split_and_threshold_set(city_df, splits):
    table, _ = sensitivity_sets()
    out = run_sensitivity(city_df, {}, splits, ["global"], ["dummy"])
    assert len(out) == len(table) * len(splits)
    assert list(out.columns[:3]) == ["threshold", "feature_set", "dimensions"]
    assert set(out.threshold) == set(THRESHOLDS)  # includes the main one (50)
    assert "completo" not in set(out.feature_set)


def test_each_distinct_set_is_run_once(city_df, splits, monkeypatch):
    calls = []
    real = sensitivity_run.run_experiments
    monkeypatch.setattr(
        sensitivity_run, "run_experiments", lambda *a, **k: calls.append(a[2]) or real(*a, **k)
    )
    run_sensitivity(city_df, {}, splits, ["global"], ["dummy"])
    assert len(calls) == len(sensitivity_sets()[1])


def test_duplicates_share_the_results(city_df, splits):
    out = run_sensitivity(city_df, {}, splits, ["global"], ["dummy"])
    at50 = out[out.threshold == 50]
    calle, comb = (
        at50[at50.feature_set == f].reset_index(drop=True) for f in ("calle", "combinado")
    )
    pd.testing.assert_series_equal(calle.dimensions, comb.dimensions)
    pd.testing.assert_series_equal(calle.f1_macro, comb.f1_macro)


def test_a_tuned_model_without_hyperparameters_for_its_set_raises_instead_of_falling_back(
    city_df, splits
):
    # logistic_regression tuned for completo and calle@50 only: the other sets have no entry
    best = {}
    for s in splits:
        best[tuning_key(s, "completo", "logistic_regression")] = {"C": 0.1}
        best[tuning_key(s, "calle", "logistic_regression", 50)] = {"C": 1}
    with pytest.raises(KeyError):
        run_sensitivity(city_df, best, splits, ["global"], ["logistic_regression"])


def test_used_fallback_is_false_in_every_row_when_every_set_has_its_own_tuning(city_df, splits):
    table, _ = sensitivity_sets()
    best = {
        tuning_key(s, r.feature_set, "logistic_regression", r.threshold): {"C": 1}
        for s in splits
        for r in table.itertuples()
    }
    out = run_sensitivity(city_df, best, splits, ["global"], ["dummy", "logistic_regression"])
    assert "used_fallback" in out.columns and not out.used_fallback.any()


def test_cli_refuses_to_overwrite(tmp_path):
    out = tmp_path / "s.csv"
    out.write_text("x")
    result = CliRunner().invoke(app, ["--output", str(out)])
    assert result.exit_code == 1 and out.read_text() == "x"


def test_cli_missing_best_params(tmp_path):
    result = CliRunner().invoke(
        app, ["--output", str(tmp_path / "s.csv"), "--best-params-path", str(tmp_path / "no.json")]
    )
    assert result.exit_code == 1
