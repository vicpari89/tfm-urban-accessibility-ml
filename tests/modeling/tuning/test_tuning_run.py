import os

import pandas as pd
import pytest
from typer.testing import CliRunner

from urban_accessibility_ml.modeling.tuning import tuning_run
from urban_accessibility_ml.modeling.tuning.tuning_constants import SEARCH_SPACES
from urban_accessibility_ml.modeling.tuning.tuning_results import load_best_params, tuning_key
from urban_accessibility_ml.modeling.tuning.tuning_run import (
    BEST_PARAMS_FILE,
    CANDIDATES_FILE,
    DEFAULT_N_JOBS,
    app,
    run_tuning,
)
from urban_accessibility_ml.preprocessing.split import build_all_splits


@pytest.fixture
def fake_partition(monkeypatch):
    """tune_partition replaced by a fast fake that records (partition, feature set, n_jobs)."""
    calls = []

    def fake(df, split, feature_set, n_jobs=1):
        calls.append((split.key, feature_set, n_jobs))
        table = pd.DataFrame(
            {
                "strategy": split.strategy,
                "name": split.name,
                "fold": split.fold,
                "model": list(SEARCH_SPACES),
                "params": [{"C": 1}] * len(SEARCH_SPACES),
                "selected": True,
            }
        )
        return {m: {"C": 1} for m in SEARCH_SPACES}, table

    monkeypatch.setattr(tuning_run, "tune_partition", fake)
    return calls


@pytest.fixture
def files(tmp_path):
    return tmp_path / "best_params.json", tmp_path / "candidates.csv"


class TestRunTuning:
    def test_feature_sets_go_in_order_over_all_partitions(self, city_df, fake_partition, files):
        run_tuning(city_df, *files)
        keys = [s.key for s in build_all_splits(city_df)]
        assert [(k, fs) for k, fs, _ in fake_partition] == [
            (k, fs) for fs in ["completo", "cartografico_aereo", "calle"] for k in keys
        ]

    def test_combinado_is_not_tuned_again_because_it_shares_calle(
        self, city_df, fake_partition, files
    ):
        run_tuning(city_df, *files)
        assert "combinado" not in {fs for _, fs, _ in fake_partition}
        key = tuning_key(build_all_splits(city_df)[0], "combinado", "random_forest")
        assert key in load_best_params(files[0])

    def test_writes_the_two_result_files(self, city_df, fake_partition, files):
        run_tuning(city_df, *files)
        n_splits = len(build_all_splits(city_df))
        assert len(load_best_params(files[0])) == 3 * n_splits * len(SEARCH_SPACES)
        assert set(pd.read_csv(files[1])["feature_set"]) == {"completo", "cartografico_aereo", "calle"}

    def test_continues_where_a_stopped_run_was(self, city_df, fake_partition, files, monkeypatch):
        real = tuning_run.tune_partition

        def stop_at_the_fourth(df, split, feature_set, n_jobs=1):
            if len(fake_partition) == 3:
                raise KeyboardInterrupt
            return real(df, split, feature_set, n_jobs)

        monkeypatch.setattr(tuning_run, "tune_partition", stop_at_the_fourth)
        with pytest.raises(KeyboardInterrupt):
            run_tuning(city_df, *files)
        assert len(load_best_params(files[0])) == 3 * len(SEARCH_SPACES)

        monkeypatch.setattr(tuning_run, "tune_partition", real)
        fake_partition.clear()
        run_tuning(city_df, *files)
        assert len(fake_partition) == 3 * len(build_all_splits(city_df)) - 3

    def test_a_second_run_tunes_nothing(self, city_df, fake_partition, files):
        run_tuning(city_df, *files)
        fake_partition.clear()
        run_tuning(city_df, *files)
        assert fake_partition == []

    def test_overwrite_starts_from_scratch(self, city_df, fake_partition, files):
        run_tuning(city_df, *files)
        fake_partition.clear()
        run_tuning(city_df, *files, overwrite=True)
        assert len(fake_partition) == 3 * len(build_all_splits(city_df))

    def test_passes_n_jobs(self, city_df, fake_partition, files):
        run_tuning(city_df, *files, n_jobs=3)
        assert {n for _, _, n in fake_partition} == {3}

    def test_default_n_jobs_is_half_of_the_cores_at_least_one(self):
        assert 1 <= DEFAULT_N_JOBS <= max(1, (os.cpu_count() or 2) // 2)

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

    @pytest.mark.parametrize(
        "args, overwrite, n_jobs",
        [([], False, DEFAULT_N_JOBS), (["--overwrite", "--n-jobs", "2"], True, 2)],
    )
    def test_reads_the_dataset_and_passes_the_options(
        self, dataset, monkeypatch, args, overwrite, n_jobs
    ):
        calls = []
        monkeypatch.setattr(
            tuning_run,
            "run_tuning",
            lambda df, overwrite, n_jobs: calls.append((len(df), overwrite, n_jobs)),
        )
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert calls == [(len(dataset), overwrite, n_jobs)]

    def test_fails_with_exit_code_1_if_the_dataset_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tuning_run, "UNIFIED_FILE", tmp_path / "does_not_exist.csv")
        assert CliRunner().invoke(app, []).exit_code == 1
