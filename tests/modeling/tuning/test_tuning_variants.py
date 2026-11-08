import pandas as pd
import pytest
from typer.testing import CliRunner

from urban_accessibility_ml.modeling.tuning import tuning_variants
from urban_accessibility_ml.modeling.tuning.tuning_results import load_best_params, save_best_params
from urban_accessibility_ml.modeling.tuning.tuning_variants import (
    TREE_MODELS,
    VARIANTS,
    app,
    run_variants,
)

MAIN = {
    "p/n/0|cruces|logistic_regression": {"C": 1},
    "p/n/0|cruces|random_forest": {"max_depth": 3},
}


@pytest.fixture
def main_file(tmp_path):
    return save_best_params(MAIN, tmp_path / "main.json")


@pytest.fixture
def fakes(monkeypatch):
    """run_tuning and run_experiments replaced by fast fakes that record their calls."""
    calls = {"tuning": [], "experiments": []}

    def fake_tuning(df, best, candidates, overwrite, n_jobs, dry_run, **options):
        calls["tuning"].append((best.name, overwrite, n_jobs, dry_run, options))
        if not dry_run:
            save_best_params({"p/n/0|cruces|random_forest": {"max_depth": 9}}, best)

    def fake_experiments(df, splits, hyperparameters):
        calls["experiments"].append(hyperparameters)
        return pd.DataFrame({"model": ["dummy"], "f1_macro": [0.5]})

    monkeypatch.setattr(tuning_variants, "run_tuning", fake_tuning)
    monkeypatch.setattr(tuning_variants, "run_experiments", fake_experiments)
    return calls


class TestRunVariants:
    def test_each_variant_is_tuned_with_its_options_on_the_tree_models(
        self, city_df, fakes, main_file, tmp_path
    ):
        run_variants(city_df, None, main_file, tmp_path / "out", n_jobs=3)
        assert [c[0] for c in fakes["tuning"]] == [f"best_params_{n}.json" for n in VARIANTS]
        for (_, _, n_jobs, _, options), name in zip(fakes["tuning"], VARIANTS):
            assert n_jobs == 3
            assert options == {"models": TREE_MODELS, **VARIANTS[name]}

    def test_writes_one_matrix_per_variant(self, city_df, fakes, main_file, tmp_path):
        run_variants(city_df, None, main_file, tmp_path / "out")
        assert sorted(p.name for p in (tmp_path / "out").glob("experiment-matrix_*.csv")) == sorted(
            f"experiment-matrix_{n}.csv" for n in VARIANTS
        )

    def test_the_trees_come_from_the_variant_and_the_rest_from_the_main_tuning(
        self, city_df, fakes, main_file, tmp_path
    ):
        run_variants(city_df, ["seed2"], main_file, tmp_path / "out")
        assert fakes["experiments"] == [
            {
                "p/n/0|cruces|logistic_regression": {"C": 1},
                "p/n/0|cruces|random_forest": {"max_depth": 9},
            }
        ]

    def test_a_second_run_does_not_run_the_experiments_again(
        self, city_df, fakes, main_file, tmp_path
    ):
        run_variants(city_df, ["seed2"], main_file, tmp_path / "out")
        run_variants(city_df, ["seed2"], main_file, tmp_path / "out")
        assert len(fakes["experiments"]) == 1

    def test_overwrite_runs_them_again_and_is_passed_to_the_tuning(
        self, city_df, fakes, main_file, tmp_path
    ):
        run_variants(city_df, ["seed2"], main_file, tmp_path / "out")
        run_variants(city_df, ["seed2"], main_file, tmp_path / "out", overwrite=True)
        assert len(fakes["experiments"]) == 2
        assert fakes["tuning"][-1][1] is True

    def test_dry_run_only_estimates(self, city_df, fakes, main_file, tmp_path):
        run_variants(city_df, None, main_file, tmp_path / "out", dry_run=True)
        assert all(c[3] for c in fakes["tuning"])
        assert fakes["experiments"] == []
        assert not (tmp_path / "out").exists()

    def test_unknown_variant_fails_before_tuning(self, city_df, fakes, main_file, tmp_path):
        with pytest.raises(ValueError, match="nope"):
            run_variants(city_df, ["nope"], main_file, tmp_path / "out")
        assert fakes["tuning"] == []

    def test_missing_main_tuning_fails_before_tuning(self, city_df, fakes, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_variants(city_df, None, tmp_path / "missing.json", tmp_path / "out")
        assert fakes["tuning"] == []

    def test_the_variants_only_tune_the_random_searches(self):
        assert "logistic_regression" not in TREE_MODELS and TREE_MODELS


class TestCommand:
    @pytest.fixture
    def dataset(self, city_df, tmp_path, monkeypatch):
        path = tmp_path / "streets.csv"
        city_df.reset_index(drop=True).to_csv(path, index=False)
        monkeypatch.setattr(tuning_variants, "UNIFIED_FILE", path)

    def test_passes_the_options(self, dataset, monkeypatch):
        calls = []
        monkeypatch.setattr(
            tuning_variants, "run_variants", lambda df, names, **kw: calls.append((names, kw))
        )
        args = ["--variant", "seed2", "--overwrite", "--n-jobs", "2", "--dry-run"]
        assert CliRunner().invoke(app, args).exit_code == 0
        assert calls == [(["seed2"], {"overwrite": True, "n_jobs": 2, "dry_run": True})]

    def test_by_default_runs_all_the_variants(self, dataset, monkeypatch):
        calls = []
        monkeypatch.setattr(
            tuning_variants, "run_variants", lambda df, names, **kw: calls.append(names)
        )
        assert CliRunner().invoke(app, []).exit_code == 0
        assert calls == [None]

    def test_fails_with_exit_code_1_on_a_known_error(self, dataset, monkeypatch):
        def fail(df, names, **kw):
            raise ValueError("Unknown variant")

        monkeypatch.setattr(tuning_variants, "run_variants", fail)
        assert CliRunner().invoke(app, []).exit_code == 1
