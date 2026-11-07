import pytest
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

    def test_tunable_defaults_are_the_sklearn_ones(self):
        model = model_random_forest.build_model()
        assert model.max_depth is None
        assert model.min_samples_leaf == 1
        assert model.max_features == "sqrt"

    def test_tunable_params_are_applied(self):
        model = model_random_forest.build_model(max_depth=3, min_samples_leaf=5, max_features=0.5)
        assert model.max_depth == 3
        assert model.min_samples_leaf == 5
        assert model.max_features == 0.5

    def test_tuning_keeps_the_fixed_params(self):
        model = model_random_forest.build_model(OTHER_SEED, max_depth=3)
        assert model.n_estimators == model_random_forest.N_ESTIMATORS
        assert model.class_weight == model_random_forest.CLASS_WEIGHT
        assert model.random_state == OTHER_SEED

    def test_max_depth_limits_the_trees(self, xy):
        X, y = xy
        forest = model_random_forest.build_model(max_depth=2).fit(X, y)
        assert max(tree.get_depth() for tree in forest.estimators_) <= 2

    def test_n_estimators_is_not_tunable(self):
        with pytest.raises(TypeError):
            model_random_forest.build_model(n_estimators=10)
