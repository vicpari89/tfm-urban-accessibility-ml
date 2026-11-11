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


def test_used_fallback_true_only_for_tuned_models_without_own_entry(city_df, splits):
    # tuned logistic_regression for calle@50 and completo only; the rest must fall back
    best = {}
    for s in splits:
        best[tuning_key(s, "completo", "logistic_regression")] = {"C": 0.1}
        best[tuning_key(s, "calle", "logistic_regression", 50)] = {"C": 1}
    out = run_sensitivity(city_df, best, splits, ["global"], ["dummy", "logistic_regression"])
    lr = out[out.model == "logistic_regression"]
    own = lr[(lr.threshold == 50) & (lr.feature_set.isin(["calle", "combinado"]))]
    assert not own.used_fallback.any()
    assert lr[~lr.index.isin(own.index)].used_fallback.all()
    assert not out[out.model == "dummy"].used_fallback.any()


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
