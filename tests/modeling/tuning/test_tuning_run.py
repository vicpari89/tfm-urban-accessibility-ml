import json
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
    sensitivity_units,
    tuning_units,
)
from urban_accessibility_ml.preprocessing.features import dimensions_for, sensitivity_sets
from urban_accessibility_ml.preprocessing.split import build_all_splits
from urban_accessibility_ml.config import FEATURE_SETS


@pytest.fixture
def fake_partition(monkeypatch):
    """tune_partition replaced by a fast fake that records (partition, feature set, n_jobs)."""
    class Calls(list):
        extra = []  # (seed, n_candidates, models) of each call
        thresholds = []  # threshold of each call

    calls = Calls()
    calls.extra = []
    calls.thresholds = []

    def fake(
        df, split, feature_set, n_jobs=1, seed=None, n_candidates=None, models=None, threshold=None
    ):
        calls.append((split.key, feature_set, n_jobs))
        calls.extra.append((seed, n_candidates, models))
        calls.thresholds.append(threshold)
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


class TestSensitivitySets:
    def test_units_are_the_distinct_sets_that_the_main_threshold_and_completo_do_not_have(self):
        tuned = {tuple(dimensions_for(fs)) for fs in FEATURE_SETS}
        units = sensitivity_units()
        dims = [tuple(dimensions_for(fs, t)) for fs, t in units]
        assert units and len(set(dims)) == len(dims)  # each set once
        assert not set(dims) & tuned  # none of the ones already tuned
        assert all(t != 50 for _, t in units)

    def test_every_set_of_the_sensitivity_is_tuned_by_the_main_ones_or_a_unit(self):
        table, _ = sensitivity_sets()
        covered = {tuple(dimensions_for(fs)) for fs in FEATURE_SETS}
        covered |= {tuple(dimensions_for(fs, t)) for fs, t in sensitivity_units()}
        assert set(table["dimensions"]) <= covered

    def test_units_without_sensitivity_are_the_feature_sets_of_the_main_threshold(self):
        assert tuning_units() == [(fs, None) for fs in ["completo", "cartografico_aereo", "calle", "combinado"]]
        assert tuning_units(True) == tuning_units() + sensitivity_units()

    def test_sensitivity_sets_are_tuned_after_the_others_with_their_threshold(
        self, city_df, fake_partition, files
    ):
        run_tuning(city_df, *files, sensitivity=True)
        n_splits, units = len(build_all_splits(city_df)), sensitivity_units()
        first_sensitivity = 3 * n_splits  # completo, cartografico_aereo and calle go first
        assert set(fake_partition.thresholds[:first_sensitivity]) == {None}
        assert fake_partition.thresholds[first_sensitivity:] == [t for _, t in units for _ in range(n_splits)]
        assert len(load_best_params(files[0])) == (3 + len(units)) * n_splits * len(SEARCH_SPACES)

    def test_the_candidates_of_a_sensitivity_set_do_not_replace_the_main_ones(
        self, city_df, fake_partition, files
    ):
        run_tuning(city_df, *files, sensitivity=True)
        table = pd.read_csv(files[1])
        labels = {f"{fs}@{t}" for fs, t in sensitivity_units()}
        assert set(table["feature_set"]) == {"completo", "cartografico_aereo", "calle"} | labels

    def test_the_main_tunings_are_not_touched_when_the_sensitivity_is_added_later(
        self, city_df, fake_partition, files
    ):
        run_tuning(city_df, *files)
        before = files[0].read_text()
        fake_partition.clear()
        fake_partition.thresholds.clear()
        run_tuning(city_df, *files, sensitivity=True)
        assert set(fake_partition.thresholds) == {t for _, t in sensitivity_units()}  # only the new
        saved, kept = load_best_params(files[0]), json.loads(before)
        assert all(saved[k] == v for k, v in kept.items())

    def test_a_second_run_with_sensitivity_tunes_nothing(self, city_df, fake_partition, files):
        run_tuning(city_df, *files, sensitivity=True)
        fake_partition.clear()
        run_tuning(city_df, *files, sensitivity=True)
        assert fake_partition == []

    def test_estimate_counts_the_sensitivity_sets(self, city_df, fake_partition):
        splits, indexed = build_all_splits(city_df), tuning_run._index_by_key(city_df)
        per_unit = 2.0 * 10 * len(SEARCH_SPACES)
        n_sets = 3 + len(sensitivity_units())
        seconds = estimate_seconds(
            indexed, splits, {}, 2, list(SEARCH_SPACES), units=tuning_units(True)
        )
        assert seconds == per_unit * n_sets * len(splits) / 2


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
            lambda df, overwrite, n_jobs, dry_run, sensitivity: calls.append(
                (len(df), overwrite, n_jobs, dry_run, sensitivity)
            ),
        )
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert calls == [(len(dataset), overwrite, n_jobs, dry_run, True)]  # the command tunes the sensitivity too

    def test_fails_with_exit_code_1_if_the_dataset_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tuning_run, "UNIFIED_FILE", tmp_path / "does_not_exist.csv")
        assert CliRunner().invoke(app, []).exit_code == 1
