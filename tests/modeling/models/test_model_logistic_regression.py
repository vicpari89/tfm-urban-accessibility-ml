import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from urban_accessibility_ml.modeling.models import model_logistic_regression

OTHER_SEED = 7


class TestLogisticRegression:
    def test_type_and_params(self):
        model = model_logistic_regression.build_model(OTHER_SEED)
        assert isinstance(model, LogisticRegression)
        assert model.class_weight == model_logistic_regression.CLASS_WEIGHT
        assert model.max_iter == model_logistic_regression.MAX_ITER
        assert model.random_state == OTHER_SEED

    def test_beats_the_dummy(self, xy, dummy_accuracy):
        X, y = xy
        pred = model_logistic_regression.build_model().fit(X, y).predict(X)
        assert (pred == y).mean() > dummy_accuracy

    def test_tunable_defaults_are_the_sklearn_ones(self):
        model = model_logistic_regression.build_model()
        assert model.C == 1.0
        assert model.penalty == "l2"

    def test_tunable_params_are_applied(self):
        model = model_logistic_regression.build_model(C=0.1, penalty="l1")
        assert model.C == 0.1
        assert model.penalty == "l1"

    def test_tuning_keeps_the_fixed_params(self):
        model = model_logistic_regression.build_model(OTHER_SEED, C=10, penalty="l1")
        assert model.class_weight == model_logistic_regression.CLASS_WEIGHT
        assert model.max_iter == model_logistic_regression.MAX_ITER
        assert model.random_state == OTHER_SEED

    @pytest.mark.parametrize("penalty, solver", [("l2", "lbfgs"), ("l1", "saga")])
    def test_solver_follows_the_penalty(self, penalty, solver):
        assert model_logistic_regression.build_model(penalty=penalty).solver == solver

    def test_l1_with_strong_regularization_zeroes_weights(self, xy):
        X, y = xy
        coef = model_logistic_regression.build_model(C=0.001, penalty="l1").fit(X, y).coef_
        assert (coef == 0).any()

    def test_smaller_c_shrinks_the_weights(self, xy):
        X, y = xy
        weak = model_logistic_regression.build_model(C=100).fit(X, y).coef_
        strong = model_logistic_regression.build_model(C=0.001).fit(X, y).coef_
        assert np.abs(strong).sum() < np.abs(weak).sum()

    def test_unknown_penalty_fails(self):
        with pytest.raises(KeyError):
            model_logistic_regression.build_model(penalty="elasticnet")

    def test_only_tunable_params_are_accepted_by_name(self):
        with pytest.raises(TypeError):
            model_logistic_regression.build_model(max_iter=5)
        with pytest.raises(TypeError):
            model_logistic_regression.build_model(OTHER_SEED, 0.1)  # C is keyword-only
