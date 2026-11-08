import json

import numpy as np
import pandas as pd
import pytest

from urban_accessibility_ml.modeling.tuning import tuning_search as tuning
from urban_accessibility_ml.modeling.tuning.tuning_results import (
    load_best_params,
    save_best_params,
    is_tuned,
    save_candidates,
    save_tuning,
    tuning_key,
)
from urban_accessibility_ml.preprocessing.split import Split, build_all_splits

BEST = {
    "within_city/Albacete/0": {"random_forest": {"max_depth": None, "max_features": 0.5}},
    "cross_city/Albacete->Algeciras/0": {"logistic_regression": {"C": 10, "penalty": "l2"}},
}


@pytest.fixture
def candidates():
    """The table of candidates, with numpy numbers in the params (like a random search gives)."""
    params = [
        {"max_depth": np.int64(3), "learning_rate": np.float64(0.05)},
        {"max_depth": np.int64(5), "learning_rate": np.float64(0.2)},
    ]
    return pd.DataFrame(
        {
            "strategy": "within_city",
            "name": "Albacete",
            "fold": 0,
            "model": "xgboost_balanced",
            "params": params,
            "f1_macro_mean": [0.9, 0.95],
            "f1_macro_std": [0.01, 0.02],
            "selected": [False, True],
        }
    )


class TestSaveBestParams:
    def test_round_trip_through_json(self, tmp_path):
        path = save_best_params(BEST, tmp_path / "best_params.json")
        assert json.loads(path.read_text()) == BEST  # None <-> null, and the '->' key

    def test_load_best_params_reads_what_was_saved(self, tmp_path):
        path = save_best_params(BEST, tmp_path / "best_params.json")
        assert load_best_params(path) == BEST

    def test_creates_the_folders_and_returns_the_path(self, tmp_path):
        path = save_best_params(BEST, tmp_path / "a" / "b" / "best_params.json")
        assert path == tmp_path / "a" / "b" / "best_params.json"
        assert path.exists()

    def test_writes_numpy_numbers_as_python_numbers(self, tmp_path):
        best = {"p/n/0": {"xgboost_balanced": {"max_depth": np.int64(3), "subsample": np.float64(0.7)}}}
        path = save_best_params(best, tmp_path / "best_params.json")
        assert json.loads(path.read_text()) == {
            "p/n/0": {"xgboost_balanced": {"max_depth": 3, "subsample": 0.7}}
        }

    def test_something_that_is_not_a_number_fails(self, tmp_path):
        with pytest.raises(TypeError, match="not JSON serializable"):
            save_best_params({"p/n/0": {"m": {"x": object()}}}, tmp_path / "best_params.json")


class TestSaveCandidates:
    def test_has_every_candidate_and_no_index_column(self, candidates, tmp_path):
        read = pd.read_csv(save_candidates(candidates, tmp_path / "a" / "candidates.csv"))
        assert len(read) == len(candidates)
        assert "Unnamed: 0" not in read.columns
        assert {"strategy", "name", "fold", "model", "params", "selected"} <= set(read.columns)

    def test_params_can_be_read_back_as_python_numbers(self, candidates, tmp_path):
        read = pd.read_csv(save_candidates(candidates, tmp_path / "candidates.csv"))
        assert [json.loads(p) for p in read["params"]] == [
            {"max_depth": 3, "learning_rate": 0.05},
            {"max_depth": 5, "learning_rate": 0.2},
        ]

    def test_does_not_modify_the_candidates(self, candidates, tmp_path):
        before = candidates.copy()
        save_candidates(candidates, tmp_path / "candidates.csv")
        pd.testing.assert_frame_equal(candidates, before)


class TestTuningKey:
    split = Split("within_city", "Albacete", 0, np.array([]), np.array([]))

    def test_calle_and_combinado_share_the_key(self):
        assert tuning_key(self.split, "calle", "lightgbm") == tuning_key(
            self.split, "combinado", "lightgbm"
        )

    def test_different_dimensions_or_model_or_split_differ(self):
        keys = {
            tuning_key(self.split, "completo", "lightgbm"),
            tuning_key(self.split, "calle", "lightgbm"),
            tuning_key(self.split, "calle", "random_forest"),
            tuning_key(Split("within_city", "Albacete", 1, np.array([]), np.array([])), "calle", "lightgbm"),
        }
        assert len(keys) == 4

    def test_format(self):
        assert (
            tuning_key(self.split, "cartografico_aereo", "random_forest")
            == "within_city/Albacete/0|cruces,mobiliario|random_forest"
        )


class TestSaveTuning:
    split = Split("within_city", "Albacete", 0, np.array([]), np.array([]))
    other = Split("within_city", "Albacete", 1, np.array([]), np.array([]))
    best = {"xgboost_balanced": {"max_depth": 5}}

    @pytest.fixture
    def files(self, tmp_path):
        return tmp_path / "best.json", tmp_path / "cand.csv"

    def test_is_tuned_after_saving_and_shared_between_equal_sets(self, candidates, files):
        save_tuning(self.split, "calle", self.best, candidates, *files)
        best = load_best_params(files[0])
        models = ["xgboost_balanced"]
        assert is_tuned(best, self.split, "calle", models)
        assert is_tuned(best, self.split, "combinado", models)  # same dimensions
        assert not is_tuned(best, self.split, "completo", models)
        assert not is_tuned(best, self.other, "calle", models)
        assert not is_tuned(best, self.split, "calle", models + ["lightgbm"])

    def test_keeps_what_was_already_saved(self, candidates, files):
        save_tuning(self.split, "calle", self.best, candidates, *files)
        save_tuning(self.other, "calle", self.best, candidates.assign(fold=1), *files)
        assert len(load_best_params(files[0])) == 2
        assert pd.read_csv(files[1])["fold"].tolist() == [0, 0, 1, 1]

    def test_saving_the_same_one_again_does_not_duplicate_candidates(self, candidates, files):
        for _ in range(2):
            save_tuning(self.split, "calle", self.best, candidates, *files)
        table = pd.read_csv(files[1])
        assert len(table) == 2
        assert set(table["feature_set"]) == {"calle"}
        assert json.loads(table["params"][0]) == {"max_depth": 3, "learning_rate": 0.05}

    def test_leaves_no_temporary_file(self, candidates, files):
        save_tuning(self.split, "calle", self.best, candidates, *files)
        assert sorted(p.name for p in files[0].parent.iterdir()) == ["best.json", "cand.csv"]
