import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling import tuning
from urban_accessibility_ml.modeling.models import build_model
from urban_accessibility_ml.modeling.search_spaces_constants import SEARCH_SPACES
from urban_accessibility_ml.modeling.tuning import _pick_best, tune_model

N_ROWS_PER_CLASS = {0: 60, 1: 45, 2: 15}  # C, B, A: imbalanced like the real data
N_FEATURES = 3
ROWS_PER_STREET = 3
DATA_SEED = 0
UNKNOWN_MODEL = "does_not_exist"
N_CANDIDATES = 16  # 8 values of C x 2 penalties
N_INNER_FOLDS = 3
TOLERANCE = 0.005
SIMPLE = {"C": 0.1, "penalty": "l1", "solver": "saga"}
COMPLEX = {"C": 10, "penalty": "l2", "solver": "lbfgs"}


@pytest.fixture
def data():
    """Small imbalanced problem: X, y and one street for each ROWS_PER_STREET consecutive rows."""
    rng = np.random.default_rng(DATA_SEED)
    y = np.concatenate([np.full(n, c) for c, n in N_ROWS_PER_CLASS.items()])
    X = pd.DataFrame(rng.normal(size=(len(y), N_FEATURES)) + y[:, None] * 0.5)
    groups = pd.Series(np.arange(len(y)) // ROWS_PER_STREET)
    return X, pd.Series(y), groups


class TestSearchSpace:
    def test_logistic_regression_is_a_grid_of_c_and_penalty(self):
        spec = SEARCH_SPACES["logistic_regression"]
        assert spec["kind"] == "grid"
        c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        assert spec["space"] == [
            {"C": c_values, "penalty": ["l2"], "solver": ["lbfgs"]},
            {"C": c_values, "penalty": ["l1"], "solver": ["saga"]},
        ]

    def test_every_candidate_is_accepted_by_the_builder(self):
        candidates = list(ParameterGrid(SEARCH_SPACES["logistic_regression"]["space"]))
        assert len(candidates) == N_CANDIDATES
        for params in candidates:
            build_model("logistic_regression", params=params)


class TestTuneModel:
    def test_unknown_model(self, data):
        with pytest.raises(ValueError, match="No search space"):
            tune_model(UNKNOWN_MODEL, *data)

    def test_table_has_every_candidate(self, data):
        _, table = tune_model("logistic_regression", *data)
        assert len(table) == N_CANDIDATES
        assert {"params", "f1_macro_mean", "f1_macro_std", "selected"} <= set(table.columns)

    def test_returns_the_selected_row(self, data):
        best, table = tune_model("logistic_regression", *data)
        assert table["selected"].sum() == 1
        assert table.loc[table["selected"], "params"].iloc[0] == best

    def test_selected_has_the_best_mean_within_the_tolerance(self, data):
        _, table = tune_model("logistic_regression", *data)
        selected_mean = table.loc[table["selected"], "f1_macro_mean"].iloc[0]
        assert selected_mean >= table["f1_macro_mean"].max() - TOLERANCE

    def test_reproducible_with_the_same_seed(self, data):
        first = tune_model("logistic_regression", *data)
        second = tune_model("logistic_regression", *data)
        assert first[0] == second[0]
        pd.testing.assert_frame_equal(first[1], second[1])

    def test_inner_folds_use_the_project_seed_and_the_streets(self, data, monkeypatch):
        X, y, groups = data
        created, split_groups = [], []

        class Recorder(StratifiedGroupKFold):
            def __init__(self, **kwargs):
                created.append(kwargs)
                super().__init__(**kwargs)

            def split(self, X, y=None, groups=None):
                split_groups.append(groups)
                return super().split(X, y, groups)

        monkeypatch.setattr(tuning, "StratifiedGroupKFold", Recorder)
        tune_model("logistic_regression", X, y, groups)
        assert created == [{"n_splits": N_INNER_FOLDS, "shuffle": True, "random_state": SEED}]
        assert all(g is not None and (np.asarray(g) == groups.to_numpy()).all() for g in split_groups)


class TestPickBest:
    @staticmethod
    def results(*scores_and_params):
        scores, params = zip(*scores_and_params)
        return {"mean_test_score": np.array(scores), "params": list(params)}

    def test_tie_picks_the_simplest(self):
        results = self.results((0.5, COMPLEX), (0.5, SIMPLE), (0.5, {"C": 0.1, "penalty": "l2", "solver": "lbfgs"}))
        assert _pick_best("logistic_regression", results) == 1  # smallest C, l1 at the same C

    def test_difference_above_the_tolerance_is_not_a_tie(self):
        results = self.results((0.6, COMPLEX), (0.5, SIMPLE))
        assert _pick_best("logistic_regression", results) == 0

    @pytest.mark.parametrize(
        "gap, complex_wins", [(TOLERANCE - 0.001, False), (TOLERANCE + 0.001, True)]
    )
    def test_tolerance_boundary(self, gap, complex_wins):
        results = self.results((0.5 + gap, COMPLEX), (0.5, SIMPLE))
        assert (_pick_best("logistic_regression", results) == 0) == complex_wins


class TestRareClass:
    def test_fold_without_class_a_in_validation_does_not_fail(self, data):
        X, y, groups = data
        # Only 3 rows of class A, all in one street: two of the three inner folds never see it
        y = y.copy()
        y[y == 2] = 1
        y.iloc[-3:] = 2
        best, table = tune_model("logistic_regression", X, y, groups)
        assert table["f1_macro_mean"].notna().all()
        assert best in list(table["params"])

    def test_candidate_that_never_predicts_class_a_scores_lower(self):
        # Separable classes: a candidate with all the weights at 0 never predicts class A
        y = pd.Series(np.repeat([0, 1, 2], [30, 30, 12]))
        noise = np.random.default_rng(DATA_SEED).normal(size=(len(y), 2)) * 0.1
        X = pd.DataFrame(noise + y.to_numpy()[:, None])
        groups = pd.Series(np.arange(len(y)) // ROWS_PER_STREET)
        _, table = tune_model("logistic_regression", X, y, groups)
        mean = table.set_index(table["params"].map(lambda p: (p["C"], p["penalty"])))["f1_macro_mean"]
        assert mean[(100, "l2")] > 0.9
        assert mean[(0.001, "l1")] < mean[(100, "l2")]
