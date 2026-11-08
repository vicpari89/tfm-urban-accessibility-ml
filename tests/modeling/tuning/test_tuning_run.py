import os
import warnings

import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from typer.testing import CliRunner

from urban_accessibility_ml.modeling.tuning import tuning_run
from urban_accessibility_ml.modeling.tuning.tuning_constants import SEARCH_SPACES
from urban_accessibility_ml.modeling.tuning.tuning_results import load_best_params, tuning_key
from urban_accessibility_ml.modeling.tuning.tuning_run import (
    BEST_PARAMS_FILE,
    CANDIDATES_FILE,
    DEFAULT_N_JOBS,
    estimate_seconds,
    app,
    run_tuning,
)
from urban_accessibility_ml.preprocessing.split import build_all_splits


@pytest.fixture
def fake_partition(monkeypatch):
    """tune_partition replaced by a fast fake that records (partition, feature set, n_jobs)."""
    class Calls(list):
        extra = []  # (seed, n_candidates, models) of each call

    calls = Calls()
    calls.extra = []

    def fake(df, split, feature_set, n_jobs=1, seed=None, n_candidates=None, models=None):
        calls.append((split.key, feature_set, n_jobs))
        calls.extra.append((seed, n_candidates, models))
        models = list(SEARCH_SPACES) if models is None else models
        table = pd.DataFrame(
            {
                "strategy": split.strategy,
                "name": split.name,
                "fold": split.fold,
                "model": models,
                "params": [{"C": 1}] * len(models),
                "selected": True,
            }
        )
        return {m: {"C": 1} for m in models}, table

    monkeypatch.setattr(tuning_run, "tune_partition", fake)
    monkeypatch.setattr(tuning_run, "time_candidate", lambda *a: (2.0, 10))
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

        def stop_at_the_fourth(*args):
            if len(fake_partition) == 3:
                raise KeyboardInterrupt
            return real(*args)

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

    def test_estimate_is_the_time_of_the_pending_tunings_divided_by_n_jobs(
        self, city_df, fake_partition, files
    ):
        splits = build_all_splits(city_df)
        indexed = tuning_run._index_by_key(city_df)
        per_unit = 2.0 * 10 * len(SEARCH_SPACES)
        assert estimate_seconds(indexed, splits, {}, 2, list(SEARCH_SPACES)) == per_unit * 3 * len(splits) / 2

    def test_estimate_is_zero_when_everything_is_tuned(self, city_df, fake_partition, files):
        run_tuning(city_df, *files)
        splits = build_all_splits(city_df)
        indexed = tuning_run._index_by_key(city_df)
        assert estimate_seconds(
            indexed, splits, load_best_params(files[0]), 2, list(SEARCH_SPACES)
        ) == 0.0

    def test_dry_run_estimates_but_tunes_and_writes_nothing(self, city_df, fake_partition, files):
        run_tuning(city_df, *files, dry_run=True)
        assert fake_partition == []
        assert not files[0].exists()

    def test_convergence_warnings_are_silenced(self, city_df, fake_partition, files):
        run_tuning(city_df, *files, dry_run=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("did not converge", ConvergenceWarning)
        assert caught == []

    def test_passes_seed_candidates_and_models_and_only_tunes_those_models(
        self, city_df, fake_partition, files
    ):
        run_tuning(city_df, *files, seed=7, n_candidates=5, models=["lightgbm"])
        assert fake_partition.extra and all(e == (7, 5, ["lightgbm"]) for e in fake_partition.extra)
        assert set(k.split("|")[2] for k in load_best_params(files[0])) == {"lightgbm"}
        fake_partition.clear()
        run_tuning(city_df, *files, models=["lightgbm"])
        assert fake_partition == []  # the saved lightgbm tunings count as done

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
        "args, overwrite, n_jobs, dry_run",
        [
            ([], False, DEFAULT_N_JOBS, False),
            (["--overwrite", "--n-jobs", "2", "--dry-run"], True, 2, True),
        ],
    )
    def test_reads_the_dataset_and_passes_the_options(
        self, dataset, monkeypatch, args, overwrite, n_jobs, dry_run
    ):
        calls = []
        monkeypatch.setattr(
            tuning_run,
            "run_tuning",
            lambda df, overwrite, n_jobs, dry_run: calls.append((len(df), overwrite, n_jobs, dry_run)),
        )
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert calls == [(len(dataset), overwrite, n_jobs, dry_run)]

    def test_fails_with_exit_code_1_if_the_dataset_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tuning_run, "UNIFIED_FILE", tmp_path / "does_not_exist.csv")
        assert CliRunner().invoke(app, []).exit_code == 1
