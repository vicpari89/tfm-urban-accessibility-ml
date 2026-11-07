import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from urban_accessibility_ml.modeling import experiments_run
from urban_accessibility_ml.modeling.experiments_run import (
    BEST_PARAMS_FILE,
    DEFAULT_FILE,
    TUNED_FILE,
    app,
    run_matrix,
)
from urban_accessibility_ml.preprocessing.split import build_all_splits

BEST = {"within_city/Albacete/0": {"logistic_regression": {"C": 10}}}


@pytest.fixture
def best_params_file(tmp_path):
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps(BEST))
    return path


@pytest.fixture
def fake_run_experiments(monkeypatch):
    """run_experiments replaced by a fast fake that records its calls."""
    calls = []

    def fake(df, splits, hyperparameters=None):
        calls.append({"splits": splits, "hyperparameters": hyperparameters})
        return pd.DataFrame({"model": ["dummy"], "f1_macro": [0.5]})

    monkeypatch.setattr(experiments_run, "run_experiments", fake)
    return calls


class TestRunMatrix:
    def test_writes_the_two_csv_files(
        self, city_df, fake_run_experiments, best_params_file, tmp_path
    ):
        default, tuned = tmp_path / "default.csv", tmp_path / "tuned.csv"
        run_matrix(city_df, best_params_file, default, tuned)
        assert len(pd.read_csv(default)) == 1
        assert len(pd.read_csv(tuned)) == 1

    def test_default_run_has_no_hyperparameters_and_tuned_run_has_the_tuned_ones(
        self, city_df, fake_run_experiments, best_params_file, tmp_path
    ):
        run_matrix(city_df, best_params_file, tmp_path / "d.csv", tmp_path / "t.csv")
        assert [c["hyperparameters"] for c in fake_run_experiments] == [None, BEST]

    def test_both_runs_use_the_same_partitions(
        self, city_df, fake_run_experiments, best_params_file, tmp_path
    ):
        run_matrix(city_df, best_params_file, tmp_path / "d.csv", tmp_path / "t.csv")
        keys = [[s.key for s in c["splits"]] for c in fake_run_experiments]
        assert keys == [[s.key for s in build_all_splits(city_df)]] * 2

    @pytest.mark.parametrize("existing", ["default.csv", "tuned.csv"])
    def test_refuses_to_overwrite_and_does_not_run(
        self, city_df, fake_run_experiments, best_params_file, tmp_path, existing
    ):
        (tmp_path / existing).write_text("keep me")
        with pytest.raises(FileExistsError, match="--overwrite"):
            run_matrix(city_df, best_params_file, tmp_path / "default.csv", tmp_path / "tuned.csv")
        assert fake_run_experiments == []  # checked before the long run
        assert (tmp_path / existing).read_text() == "keep me"

    def test_overwrites_when_asked(
        self, city_df, fake_run_experiments, best_params_file, tmp_path
    ):
        default, tuned = tmp_path / "default.csv", tmp_path / "tuned.csv"
        default.write_text("old")
        tuned.write_text("old")
        run_matrix(city_df, best_params_file, default, tuned, overwrite=True)
        assert len(pd.read_csv(default)) == 1

    def test_missing_best_params_fails_before_running(
        self, city_df, fake_run_experiments, tmp_path
    ):
        with pytest.raises(FileNotFoundError):
            run_matrix(city_df, tmp_path / "missing.json", tmp_path / "d.csv", tmp_path / "t.csv")
        assert fake_run_experiments == []

    def test_default_files_are_in_reports_results(self):
        assert BEST_PARAMS_FILE.parent == DEFAULT_FILE.parent == TUNED_FILE.parent
        assert BEST_PARAMS_FILE.parent.name == "results"


class TestCommand:
    @pytest.fixture
    def dataset(self, city_df, tmp_path, monkeypatch):
        path = tmp_path / "streets.csv"
        city_df.reset_index(drop=True).to_csv(path, index=False)
        monkeypatch.setattr(experiments_run, "UNIFIED_FILE", path)
        return city_df

    @pytest.mark.parametrize("args, overwrite", [([], False), (["--overwrite"], True)])
    def test_reads_the_dataset_and_passes_overwrite(self, dataset, monkeypatch, args, overwrite):
        calls = []
        monkeypatch.setattr(
            experiments_run,
            "run_matrix",
            lambda df, overwrite=False: calls.append((len(df), overwrite)),
        )
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert calls == [(len(dataset), overwrite)]

    @pytest.mark.parametrize(
        "error", [FileExistsError("Already exist"), FileNotFoundError("best_params.json")]
    )
    def test_fails_with_exit_code_1_if_a_file_exists_or_is_missing(
        self, dataset, monkeypatch, error
    ):
        def fail(df, overwrite=False):
            raise error

        monkeypatch.setattr(experiments_run, "run_matrix", fail)
        assert CliRunner().invoke(app, []).exit_code == 1

    def test_fails_with_exit_code_1_if_the_dataset_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(experiments_run, "UNIFIED_FILE", tmp_path / "does_not_exist.csv")
        assert CliRunner().invoke(app, []).exit_code == 1
