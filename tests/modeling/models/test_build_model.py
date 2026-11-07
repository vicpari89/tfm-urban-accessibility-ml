import numpy as np
import pytest

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling.models import BUILDERS, NEED_SAMPLE_WEIGHT, build_model

MODEL_NAMES = ["dummy", "logistic_regression", "random_forest", "xgboost_balanced", "xgboost_unbalanced", "lightgbm"]
OTHER_SEED = 7
UNKNOWN_MODEL = "does_not_exist"
CLASSES = {0, 1, 2}  # C, B, A


class TestBuildModel:
    def test_builders_names(self):
        assert list(BUILDERS) == MODEL_NAMES

    def test_need_sample_weight_has_every_model(self):
        assert list(NEED_SAMPLE_WEIGHT) == MODEL_NAMES

    def test_only_xgboost_balanced_needs_sample_weight(self):
        assert [name for name, needs in NEED_SAMPLE_WEIGHT.items() if needs] == ["xgboost_balanced"]

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(UNKNOWN_MODEL)

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_builds_a_new_model_each_call(self, name):
        assert build_model(name) is not build_model(name)

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_default_seed_is_config_seed(self, name):
        assert build_model(name).get_params().get("random_state") == SEED

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_fit_predict_returns_encoded_classes(self, name, xy):
        X, y = xy
        pred = build_model(name).fit(X, y).predict(X)
        assert pred.shape == y.shape
        assert set(np.unique(pred)) <= CLASSES

    @pytest.mark.parametrize("name", MODEL_NAMES[1:])
    def test_same_seed_same_predictions(self, name, xy):
        X, y = xy
        first = build_model(name, seed=OTHER_SEED).fit(X, y).predict(X)
        second = build_model(name, seed=OTHER_SEED).fit(X, y).predict(X)
        assert (first == second).all()

    def test_builders_and_need_sample_weight_have_the_same_models(self):
        assert list(BUILDERS) == list(NEED_SAMPLE_WEIGHT)


class TestBuildModelParams:
    PARAMS = {
        "logistic_regression": {"C": 0.1, "penalty": "l1"},
        "random_forest": {"max_depth": 3, "min_samples_leaf": 5, "max_features": 0.5},
        "xgboost_balanced": {"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 3},
        "lightgbm": {"num_leaves": 7, "min_child_samples": 8, "subsample": 0.8},
    }

    @pytest.mark.parametrize("name", PARAMS)
    def test_params_override_the_defaults(self, name):
        got = build_model(name, params=self.PARAMS[name]).get_params()
        assert all(got[k] == v for k, v in self.PARAMS[name].items())

    @pytest.mark.parametrize("name", MODEL_NAMES)
    @pytest.mark.parametrize("params", [{"random_state": OTHER_SEED}, {"does_not_exist": 1}])
    def test_unaccepted_params_fail(self, name, params):
        with pytest.raises(TypeError):
            build_model(name, params=params)

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_no_params_is_the_same_as_before(self, name):
        assert build_model(name, params=None).get_params() == build_model(name).get_params()
        assert build_model(name, params={}).get_params() == build_model(name).get_params()

    @pytest.mark.parametrize("name", ["dummy", "xgboost_unbalanced"])
    def test_models_that_are_not_tuned_accept_no_params(self, name):
        with pytest.raises(TypeError):
            build_model(name, params={"max_depth": 2})

    @pytest.mark.parametrize("penalty", ["l1", "l2"])
    def test_logistic_regression_fits_with_every_penalty(self, penalty, xy):
        X, y = xy
        pred = build_model("logistic_regression", params={"penalty": penalty}).fit(X, y).predict(X)
        assert set(np.unique(pred)) <= CLASSES

    def test_logistic_regression_unknown_penalty_fails(self):
        with pytest.raises(KeyError):
            build_model("logistic_regression", params={"penalty": "elasticnet"})

    def test_lightgbm_subsample_freq_is_set(self):
        assert build_model("lightgbm").get_params()["subsample_freq"] == 1
