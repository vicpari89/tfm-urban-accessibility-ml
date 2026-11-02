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
