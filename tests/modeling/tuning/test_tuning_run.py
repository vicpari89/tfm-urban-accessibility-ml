import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from urban_accessibility_ml.modeling.tuning import tuning_run
from urban_accessibility_ml.modeling.tuning.tuning_run import (
    BEST_PARAMS_FILE,
    CANDIDATES_FILE,
    app,
    run_tuning,
)
from urban_accessibility_ml.preprocessing.split import build_all_splits

FAKE_BEST = {"within_city/Albacete/0": {"logistic_regression": {"C": 10}}}
FAKE_CANDIDATES = pd.DataFrame({"model": ["logistic_regression"], "params": [{"C": 10}], "selected": [True]})


@pytest.fixture
def fake_tune_all(monkeypatch):
    """tune_all replaced by a fast fake that records its calls."""
    calls = []

    def fake(df, splits):
        calls.append(splits)
        return FAKE_BEST, FAKE_CANDIDATES

    monkeypatch.setattr(tuning_run, "tune_all", fake)
    return calls


class TestRunTuning:
    def test_writes_the_two_result_files(self, city_df, fake_tune_all, tmp_path):
        best, candidates = tmp_path / "best_params.json", tmp_path / "candidates.csv"
        run_tuning(city_df, best, candidates)
        assert json.loads(best.read_text()) == FAKE_BEST
        assert len(pd.read_csv(candidates)) == 1

    def test_tunes_every_partition_of_build_all_splits(self, city_df, fake_tune_all, tmp_path):
        run_tuning(city_df, tmp_path / "b.json", tmp_path / "c.csv")
        assert [s.key for s in fake_tune_all[0]] == [s.key for s in build_all_splits(city_df)]

    @pytest.mark.parametrize("existing", ["best_params.json", "candidates.csv"])
    def test_refuses_to_overwrite_and_does_not_tune(self, city_df, fake_tune_all, tmp_path, existing):
        (tmp_path / existing).write_text("keep me")
        with pytest.raises(FileExistsError, match="--overwrite"):
            run_tuning(city_df, tmp_path / "best_params.json", tmp_path / "candidates.csv")
        assert fake_tune_all == []  # checked before the long tuning
        assert (tmp_path / existing).read_text() == "keep me"

    def test_overwrites_when_asked(self, city_df, fake_tune_all, tmp_path):
        best, candidates = tmp_path / "best_params.json", tmp_path / "candidates.csv"
        best.write_text("old")
        candidates.write_text("old")
        run_tuning(city_df, best, candidates, overwrite=True)
        assert json.loads(best.read_text()) == FAKE_BEST

    def test_default_files_are_in_reports_results(self):
        assert BEST_PARAMS_FILE.parent == CANDIDATES_FILE.parent
        assert BEST_PARAMS_FILE.parent.name == "results"
        assert BEST_PARAMS_FILE.name == "best_params.json"


class TestCommand:
    @pytest.fixture
    def dataset(self, city_df, tmp_path, monkeypatch):
        path = tmp_path / "streets.csv"
        city_df.reset_index(drop=True).to_csv(path, index=False)
        monkeypatch.setattr(tuning_run, "UNIFIED_FILE", path)
        return city_df

    @pytest.mark.parametrize("args, overwrite", [([], False), (["--overwrite"], True)])
    def test_reads_the_dataset_and_passes_overwrite(self, dataset, monkeypatch, args, overwrite):
        calls = []
        monkeypatch.setattr(
            tuning_run, "run_tuning", lambda df, overwrite=False: calls.append((len(df), overwrite))
        )
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert calls == [(len(dataset), overwrite)]

    def test_fails_with_exit_code_1_if_the_files_exist(self, dataset, monkeypatch):
        def refuse(df, overwrite=False):
            raise FileExistsError("Already exist")

        monkeypatch.setattr(tuning_run, "run_tuning", refuse)
        assert CliRunner().invoke(app, []).exit_code == 1

    def test_fails_with_exit_code_1_if_the_dataset_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tuning_run, "UNIFIED_FILE", tmp_path / "does_not_exist.csv")
        assert CliRunner().invoke(app, []).exit_code == 1
