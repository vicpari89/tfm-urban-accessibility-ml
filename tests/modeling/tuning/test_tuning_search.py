import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid, ParameterSampler, StratifiedGroupKFold

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling.tuning import tuning_search as tuning
from urban_accessibility_ml.modeling.models import build_model, model_xgboost_balanced
from urban_accessibility_ml.modeling.tuning.tuning_constants import (
    SEARCH_SPACES,
    TUNING_N_CANDIDATES,
)
from urban_accessibility_ml.modeling.tuning.tuning_search import _pick_best, tune_all, tune_model, tune_partition
from urban_accessibility_ml.preprocessing.split import Split, build_all_splits

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

# The two candidates of the light logistic regression search used by the tests that need a real fit
SMALL_LR_SPACE = [
    {"C": [100], "penalty": ["l2"], "solver": ["lbfgs"]},
    {"C": [0.001], "penalty": ["l1"], "solver": ["saga"]},
]
N_SMALL_LR_CANDIDATES = 2


def light_build_model(name):
    """
    build_model for the tests: one thread (n_jobs=-1 costs more than it saves with so few rows)
    and small forests, so the fits of the tests are fast.
    """
    model = build_model(name)
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    if name == "random_forest":
        model.set_params(n_estimators=10)
    return model


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


@pytest.fixture
def small_lr(monkeypatch):
    """The logistic regression search with 2 candidates instead of the 16 of the real one."""
    spaces = {"logistic_regression": {"kind": "grid", "space": SMALL_LR_SPACE}}
    monkeypatch.setattr(tuning, "SEARCH_SPACES", spaces)
    monkeypatch.setattr(tuning, "build_model", light_build_model)


@pytest.mark.usefixtures("small_lr")
class TestTuneModel:
    def test_unknown_model(self, data):
        with pytest.raises(ValueError, match="No search space"):
            tune_model(UNKNOWN_MODEL, *data)

    def test_table_has_every_candidate(self, data):
        _, table = tune_model("logistic_regression", *data)
        assert len(table) == N_SMALL_LR_CANDIDATES
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
        same_c_l2 = {"C": 0.1, "penalty": "l2", "solver": "lbfgs"}
        results = self.results((0.5, COMPLEX), (0.5, SIMPLE), (0.5, same_c_l2))
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


@pytest.mark.usefixtures("small_lr")
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


class TestRandomForestSearch:
    SPACE = SEARCH_SPACES["random_forest"]["space"]
    RANGES = {"max_depth": (3, 15), "min_samples_leaf": (1, 10), "max_features": (0.3, 1.0)}

    @pytest.fixture
    def tiny(self, monkeypatch):
        """A small random search, to keep the tests fast: 3 candidates of small forests."""
        space = {"max_depth": [2, 3], "min_samples_leaf": [1, 5], "max_features": ["sqrt", 1.0]}
        monkeypatch.setattr(tuning, "SEARCH_SPACES", {"random_forest": {"kind": "random", "space": space}})
        monkeypatch.setattr(tuning, "TUNING_N_CANDIDATES", 3)
        monkeypatch.setattr(tuning, "build_model", light_build_model)

    def test_is_a_random_search_of_30_candidates(self):
        assert SEARCH_SPACES["random_forest"]["kind"] == "random"
        assert TUNING_N_CANDIDATES == 30

    def test_space_has_the_hyperparameters_of_the_ticket(self):
        assert set(self.SPACE) == set(self.RANGES)

    def test_30_different_candidates(self):
        candidates = list(ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED))
        assert len(candidates) == TUNING_N_CANDIDATES
        assert len({tuple(c.items()) for c in candidates}) == TUNING_N_CANDIDATES

    def test_sampled_candidates_are_inside_the_ranges(self):
        candidates = list(ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED))
        for name, (low, high) in self.RANGES.items():
            assert all(low <= c[name] <= high for c in candidates), name

    def test_integer_hyperparameters_are_sampled_as_integers(self):
        candidates = ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED)
        for c in candidates:
            assert float(c["max_depth"]).is_integer()
            assert float(c["min_samples_leaf"]).is_integer()

    def test_every_candidate_is_accepted_by_the_builder(self):
        for params in ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED):
            build_model("random_forest", params=params)

    def test_n_estimators_is_not_searched(self):
        assert "n_estimators" not in self.SPACE
        assert build_model("random_forest").n_estimators == 300

    def test_table_has_the_budget_and_one_selected(self, data, tiny):
        best, table = tune_model("random_forest", *data)
        assert len(table) == 3
        assert table["selected"].sum() == 1
        assert table.loc[table["selected"], "params"].iloc[0] == best

    def test_reproducible_with_the_same_seed(self, data, tiny):
        first = tune_model("random_forest", *data)
        second = tune_model("random_forest", *data)
        assert first[0] == second[0]
        pd.testing.assert_frame_equal(first[1], second[1])

    def test_tie_picks_the_simplest_forest(self):
        results = {
            "mean_test_score": np.array([0.5, 0.5, 0.5, 0.5]),
            "params": [
                {"max_depth": 12, "min_samples_leaf": 10, "max_features": 0.4},
                {"max_depth": 5, "min_samples_leaf": 1, "max_features": 0.4},
                {"max_depth": 5, "min_samples_leaf": 5, "max_features": 0.9},
                {"max_depth": 5, "min_samples_leaf": 5, "max_features": 0.4},
            ],
        }
        # depth 5 beats 12; then the biggest leaves (5); then the fewest variables (0.4)
        assert _pick_best("random_forest", results) == 3


class TestLightGBMSearch:
    SPACE = SEARCH_SPACES["lightgbm"]["space"]
    RANGES = {
        "learning_rate": (0.01, 0.3),
        "num_leaves": (7, 63),
        "n_estimators": (100, 500),
        "min_child_samples": (5, 20),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
    }

    @pytest.fixture
    def tiny(self, monkeypatch):
        """A small random search, to keep the tests fast: 3 candidates of a few small trees."""
        space = {"num_leaves": [7, 15], "n_estimators": [10, 20], "min_child_samples": [5, 10]}
        spaces = {"lightgbm": {"kind": "random", "space": space}}
        monkeypatch.setattr(tuning, "SEARCH_SPACES", spaces)
        monkeypatch.setattr(tuning, "TUNING_N_CANDIDATES", 3)
        monkeypatch.setattr(tuning, "build_model", light_build_model)

    def test_is_a_random_search_of_30_candidates(self):
        assert SEARCH_SPACES["lightgbm"]["kind"] == "random"
        assert TUNING_N_CANDIDATES == 30

    def test_space_has_the_hyperparameters_of_the_ticket(self):
        assert set(self.SPACE) == set(self.RANGES)

    def test_sampled_candidates_are_inside_the_ranges(self):
        candidates = list(ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED))
        assert len(candidates) == TUNING_N_CANDIDATES
        for name, (low, high) in self.RANGES.items():
            assert all(low <= c[name] <= high for c in candidates), name

    def test_integer_hyperparameters_are_sampled_as_integers(self):
        candidates = ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED)
        for c in candidates:
            for name in ["num_leaves", "n_estimators", "min_child_samples"]:
                assert float(c[name]).is_integer(), name

    def test_sampled_candidates_are_accepted_by_the_builder(self):
        for params in ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED):
            build_model("lightgbm", params=params)

    def test_learning_rate_is_sampled_on_a_log_scale(self):
        samples = ParameterSampler(self.SPACE, n_iter=200, random_state=SEED)
        assert np.median([c["learning_rate"] for c in samples]) < 0.1  # uniform: near 0.155

    def test_table_has_the_budget_and_one_selected(self, data, tiny):
        best, table = tune_model("lightgbm", *data)
        assert len(table) == 3
        assert table["selected"].sum() == 1
        assert table.loc[table["selected"], "params"].iloc[0] == best

    def test_reproducible_with_the_same_seed(self, data, tiny):
        first = tune_model("lightgbm", *data)
        second = tune_model("lightgbm", *data)
        assert first[0] == second[0]
        pd.testing.assert_frame_equal(first[1], second[1])

    def test_tie_picks_the_simplest(self):
        results = {
            "mean_test_score": np.array([0.5, 0.5, 0.5, 0.5]),
            "params": [
                {"num_leaves": 31, "n_estimators": 100, "min_child_samples": 20},
                {"num_leaves": 7, "n_estimators": 400, "min_child_samples": 5},
                {"num_leaves": 7, "n_estimators": 200, "min_child_samples": 5},
                {"num_leaves": 7, "n_estimators": 200, "min_child_samples": 15},
            ],
        }
        # 7 leaves beat 31; then fewer trees (200); then the biggest leaves (min_child_samples 15)
        assert _pick_best("lightgbm", results) == 3


class TestXGBoostSearch:
    SPACE = SEARCH_SPACES["xgboost_balanced"]["space"]
    RANGES = {
        "learning_rate": (0.01, 0.3),
        "max_depth": (2, 6),
        "n_estimators": (100, 500),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "min_child_weight": (1, 10),
    }

    @pytest.fixture
    def tiny(self, monkeypatch):
        """A small random search, to keep the tests fast: 3 candidates of a few small trees."""
        space = {"max_depth": [2, 3], "n_estimators": [10, 20], "min_child_weight": [1, 5]}
        spaces = {"xgboost_balanced": {"kind": "random", "space": space}}
        monkeypatch.setattr(tuning, "SEARCH_SPACES", spaces)
        monkeypatch.setattr(tuning, "TUNING_N_CANDIDATES", 3)
        monkeypatch.setattr(tuning, "build_model", light_build_model)

    def test_is_a_random_search_of_30_candidates(self):
        assert SEARCH_SPACES["xgboost_balanced"]["kind"] == "random"
        assert TUNING_N_CANDIDATES == 30

    def test_space_has_the_hyperparameters_of_the_ticket(self):
        assert set(self.SPACE) == set(self.RANGES)

    def test_sampled_candidates_are_inside_the_ranges(self):
        candidates = list(ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED))
        assert len(candidates) == TUNING_N_CANDIDATES
        for name, (low, high) in self.RANGES.items():
            assert all(low <= c[name] <= high for c in candidates), name

    def test_integer_hyperparameters_are_sampled_as_integers(self):
        candidates = ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED)
        for c in candidates:
            for name in ["max_depth", "n_estimators", "min_child_weight"]:
                assert float(c[name]).is_integer(), name

    def test_sampled_candidates_are_accepted_by_the_builder(self):
        for params in ParameterSampler(self.SPACE, n_iter=TUNING_N_CANDIDATES, random_state=SEED):
            build_model("xgboost_balanced", params=params)

    def test_learning_rate_is_sampled_on_a_log_scale(self):
        samples = ParameterSampler(self.SPACE, n_iter=200, random_state=SEED)
        assert np.median([c["learning_rate"] for c in samples]) < 0.1  # uniform: near 0.155

    def test_table_has_the_budget_and_one_selected(self, data, tiny):
        best, table = tune_model("xgboost_balanced", *data)
        assert len(table) == 3
        assert table["selected"].sum() == 1
        assert table.loc[table["selected"], "params"].iloc[0] == best

    def test_reproducible_with_the_same_seed(self, data, tiny):
        first = tune_model("xgboost_balanced", *data)
        second = tune_model("xgboost_balanced", *data)
        assert first[0] == second[0]
        pd.testing.assert_frame_equal(first[1], second[1])

    def test_weights_are_computed_in_each_inner_fold(self, data, tiny, monkeypatch):
        X, y, groups = data
        calls = []
        real = model_xgboost_balanced.compute_sample_weight

        def spy(class_weight, y_fit):
            calls.append(len(y_fit))
            return real(class_weight, y_fit)

        monkeypatch.setattr(model_xgboost_balanced, "compute_sample_weight", spy)
        tune_model("xgboost_balanced", X, y, groups)
        n_fits = 3 * N_INNER_FOLDS  # 3 candidates x 3 inner folds
        assert len(calls) == n_fits + 1  # plus the final refit with the best candidate
        assert all(n < len(y) for n in calls[:n_fits])  # only the rows training in that fold
        assert calls[-1] == len(y)  # the refit uses all the train rows
        # inner train sets: each row is in 2 of the 3 inner train sets of a candidate
        assert sum(calls[:N_INNER_FOLDS]) == 2 * len(y)

    def test_tie_picks_the_simplest(self):
        results = {
            "mean_test_score": np.array([0.5, 0.5, 0.5, 0.5]),
            "params": [
                {"max_depth": 6, "n_estimators": 100, "min_child_weight": 1},
                {"max_depth": 2, "n_estimators": 400, "min_child_weight": 1},
                {"max_depth": 2, "n_estimators": 200, "min_child_weight": 1},
                {"max_depth": 2, "n_estimators": 200, "min_child_weight": 8},
            ],
        }
        # depth 2 beats 6; then fewer trees (200); then the biggest min_child_weight (8)
        assert _pick_best("xgboost_balanced", results) == 3


class TestSplitKey:
    def test_key_is_strategy_name_fold(self):
        split = Split("cross_city", "Albacete->Algeciras", 0, np.array(["a"]), np.array(["b"]))
        assert split.key == "cross_city/Albacete->Algeciras/0"


@pytest.fixture
def tiny_all(monkeypatch):
    """The four models with tiny searches (2 candidates of a few small trees)."""
    # every hyperparameter that COMPLEXITY reads (the tie rule) has to be in the space
    spaces = {
        "logistic_regression": {
            "kind": "grid",
            "space": [{"C": [0.1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}],
        },
        "random_forest": {
            "kind": "random",
            "space": {"max_depth": [2, 3], "min_samples_leaf": [1, 5], "max_features": [0.5, 1.0]},
        },
        "xgboost_balanced": {
            "kind": "random",
            "space": {"max_depth": [2, 3], "n_estimators": [5, 10], "min_child_weight": [1, 5]},
        },
        "lightgbm": {
            "kind": "random",
            "space": {"num_leaves": [7, 15], "n_estimators": [5, 10], "min_child_samples": [5, 10]},
        },
    }
    monkeypatch.setattr(tuning, "SEARCH_SPACES", spaces)
    monkeypatch.setattr(tuning, "TUNING_N_CANDIDATES", 2)
    monkeypatch.setattr(tuning, "build_model", light_build_model)
    return spaces


def fake_table():
    """The table that a tune_model would return, without fitting anything."""
    return {}, pd.DataFrame({"params": [{}], "selected": [True]})


class TestTunePartition:
    def test_never_reads_the_test_rows(self, city_df, tiny_all, monkeypatch):
        split = build_all_splits(city_df)[0]
        seen = []

        def spy(model_name, X_train, y_train, groups_train):
            seen.append(set(X_train.index))
            return fake_table()  # what matters here is the rows it receives, not the fit

        monkeypatch.setattr(tuning, "tune_model", spy)
        tune_partition(city_df, split)
        assert len(seen) == len(tiny_all)
        assert all(ids == set(split.train_ids) for ids in seen)
        assert not set(split.test_ids) & set().union(*seen)

    def test_the_result_does_not_depend_on_the_test_rows(self, city_df, tiny_all):
        split = build_all_splits(city_df)[0]
        changed = city_df.copy()
        test_rows = changed.index.isin(split.test_ids)
        changed.loc[test_rows, "clase_total"] = "C"  # corrupt the test targets...
        changed.loc[test_rows, "puntuacion_pendiente_global"] = 99.0  # ...and one test input
        assert tune_partition(city_df, split)[0] == tune_partition(changed, split)[0]

    def test_returns_params_and_candidates_of_every_model(self, city_df, tiny_all):
        split = build_all_splits(city_df)[0]
        best, candidates = tune_partition(city_df, split)
        assert list(best) == list(tiny_all)
        assert set(candidates["model"]) == set(tiny_all)
        keys = ["strategy", "name", "fold"]
        assert candidates[keys].drop_duplicates().to_dict("records") == [
            {"strategy": split.strategy, "name": split.name, "fold": split.fold}
        ]
        assert candidates.groupby("model")["selected"].sum().eq(1).all()  # one selected per model


class TestTuneAll:
    @pytest.fixture
    def splits(self, city_df):
        all_splits = build_all_splits(city_df)
        return [all_splits[0], all_splits[10]]  # within_city Albacete 0, and a cross_city one

    @pytest.fixture
    def fake_partition(self, monkeypatch):
        """tune_partition without fitting: it records the dataframe it receives for each split."""
        received = []

        def fake(df, split):
            received.append((df, split.key))
            best, table = fake_table()
            return {"logistic_regression": {"C": 1}}, table.assign(name=split.name)

        monkeypatch.setattr(tuning, "tune_partition", fake)
        return received

    def test_result_has_every_partition_with_every_model_and_candidate(
        self, city_df, tiny_all, splits
    ):
        best, candidates = tune_all(city_df, splits)
        assert list(best) == [s.key for s in splits]
        assert any(k.startswith("cross_city/") and "->" in k for k in best)
        assert all(list(models) == list(tiny_all) for models in best.values())
        pairs = candidates[["strategy", "name", "fold", "model"]].drop_duplicates()
        assert len(pairs) == len(splits) * len(tiny_all)

    def test_calls_tune_partition_once_per_partition(self, city_df, splits, fake_partition):
        tune_all(city_df, splits)
        assert [key for _, key in fake_partition] == [s.key for s in splits]

    def test_gives_tune_partition_the_dataframe_indexed_by_key_tramo(
        self, city_df, splits, fake_partition
    ):
        tune_all(city_df.reset_index(drop=True), splits)  # key_tramo only as a column
        assert all(df.index.name == "key_tramo" for df, _ in fake_partition)

    def test_joins_the_tables_of_all_the_partitions(self, city_df, splits, fake_partition):
        _, candidates = tune_all(city_df, splits)
        assert list(candidates["name"]) == [s.name for s in splits]

    def test_does_not_modify_the_dataframe(self, city_df, splits, fake_partition):
        plain = city_df.reset_index(drop=True)
        before = plain.copy()
        tune_all(plain, splits)
        pd.testing.assert_frame_equal(plain, before)

    def test_is_reproducible(self, city_df, tiny_all, splits):
        first = tune_all(city_df, splits[:1])
        second = tune_all(city_df, splits[:1])
        assert first[0] == second[0]
        pd.testing.assert_frame_equal(first[1], second[1])
