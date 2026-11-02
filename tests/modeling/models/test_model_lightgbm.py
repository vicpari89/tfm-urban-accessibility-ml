from lightgbm import LGBMClassifier

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
