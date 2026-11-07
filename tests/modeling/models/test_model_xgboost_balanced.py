import pytest
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from urban_accessibility_ml.modeling.models import model_xgboost_balanced

OTHER_SEED = 7


class TestXGBoostBalanced:
    def test_type_and_params(self):
        model = model_xgboost_balanced.build_model(OTHER_SEED)
        assert isinstance(model, XGBClassifier)
        assert model.n_estimators == model_xgboost_balanced.N_ESTIMATORS
        assert model.n_jobs == model_xgboost_balanced.N_JOBS
        assert model.random_state == OTHER_SEED

    def test_has_no_class_weight(self):
        assert not hasattr(model_xgboost_balanced.build_model(), "class_weight")

    def test_needs_sample_weight(self):
        assert model_xgboost_balanced.NEED_SAMPLE_WEIGHT is True

    def test_beats_the_dummy_with_balanced_sample_weight(self, xy, dummy_accuracy):
        X, y = xy
        weights = compute_sample_weight("balanced", y)
        pred = model_xgboost_balanced.build_model().fit(X, y, sample_weight=weights).predict(X)
        assert (pred == y).mean() > dummy_accuracy

    def test_tunable_defaults_are_the_xgboost_ones(self):
        model = model_xgboost_balanced.build_model()
        assert model.n_estimators == model_xgboost_balanced.N_ESTIMATORS
        for name in ["learning_rate", "max_depth", "subsample", "colsample_bytree", "min_child_weight"]:
            assert getattr(model, name) is None  # None = the XGBoost default

    def test_tunable_params_are_applied(self):
        params = {
            "learning_rate": 0.05,
            "max_depth": 2,
            "n_estimators": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
        }
        model = model_xgboost_balanced.build_model(**params)
        assert all(getattr(model, k) == v for k, v in params.items())

    def test_tuning_keeps_the_fixed_params(self):
        model = model_xgboost_balanced.build_model(OTHER_SEED, max_depth=2)
        assert model.n_jobs == model_xgboost_balanced.N_JOBS
        assert model.random_state == OTHER_SEED

    def test_beats_the_dummy_with_tuned_params(self, xy, dummy_accuracy):
        X, y = xy
        weights = compute_sample_weight("balanced", y)
        model = model_xgboost_balanced.build_model(max_depth=2, n_estimators=50, learning_rate=0.2)
        pred = model.fit(X, y, sample_weight=weights).predict(X)
        assert (pred == y).mean() > dummy_accuracy

    def test_unknown_param_fails(self):
        with pytest.raises(TypeError):
            model_xgboost_balanced.build_model(gamma=1)
