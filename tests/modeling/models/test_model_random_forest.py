from sklearn.ensemble import RandomForestClassifier

from urban_accessibility_ml.modeling.models import model_random_forest

OTHER_SEED = 7


class TestRandomForest:
    def test_type_and_params(self):
        model = model_random_forest.build_model(OTHER_SEED)
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == model_random_forest.N_ESTIMATORS
        assert model.class_weight == model_random_forest.CLASS_WEIGHT
        assert model.n_jobs == model_random_forest.N_JOBS
        assert model.random_state == OTHER_SEED

    def test_beats_the_dummy(self, xy, dummy_accuracy):
        X, y = xy
        pred = model_random_forest.build_model().fit(X, y).predict(X)
        assert (pred == y).mean() > dummy_accuracy
