from lightgbm import LGBMClassifier
import pytest

from urban_accessibility_ml.modeling.models import model_lightgbm

OTHER_SEED = 7


class TestLightGBM:
    def test_type_and_params(self):
        model = model_lightgbm.build_model(OTHER_SEED)
        assert isinstance(model, LGBMClassifier)
        assert model.n_estimators == model_lightgbm.N_ESTIMATORS
        assert model.class_weight == model_lightgbm.CLASS_WEIGHT
        assert model.n_jobs == model_lightgbm.N_JOBS
        assert model.verbose == model_lightgbm.VERBOSITY
        assert model.random_state == OTHER_SEED

    def test_beats_the_dummy(self, xy, dummy_accuracy):
        X, y = xy
        pred = model_lightgbm.build_model().fit(X, y).predict(X)
        assert (pred == y).mean() > dummy_accuracy

    def test_tunable_defaults_are_the_lightgbm_ones(self):
        model = model_lightgbm.build_model()
        assert model.learning_rate == 0.1
        assert model.num_leaves == 31
        assert model.min_child_samples == 20
        assert model.subsample == 1.0
        assert model.colsample_bytree == 1.0

    def test_tunable_params_are_applied(self):
        params = {
            "learning_rate": 0.05,
            "num_leaves": 7,
            "n_estimators": 50,
            "min_child_samples": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
        }
        model = model_lightgbm.build_model(**params)
        assert all(getattr(model, k) == v for k, v in params.items())

    def test_tuning_keeps_the_fixed_params(self):
        model = model_lightgbm.build_model(OTHER_SEED, num_leaves=7)
        assert model.class_weight == model_lightgbm.CLASS_WEIGHT
        assert model.n_jobs == model_lightgbm.N_JOBS
        assert model.random_state == OTHER_SEED

    def test_subsample_freq_makes_subsample_work(self):
        assert model_lightgbm.build_model().subsample_freq == model_lightgbm.SUBSAMPLE_FREQ == 1

    def test_beats_the_dummy_with_tuned_params(self, xy, dummy_accuracy):
        X, y = xy
        model = model_lightgbm.build_model(num_leaves=7, min_child_samples=5, n_estimators=50)
        assert (model.fit(X, y).predict(X) == y).mean() > dummy_accuracy

    def test_unknown_param_fails(self):
        with pytest.raises(TypeError):
            model_lightgbm.build_model(max_depth=3)
