import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from urban_accessibility_ml.modeling.models import model_dummy

OTHER_SEED = 7


class TestDummy:
    def test_type_and_params(self):
        model = model_dummy.build_model(OTHER_SEED)
        assert isinstance(model, DummyClassifier)
        assert model.strategy == model_dummy.STRATEGY
        assert model.random_state == OTHER_SEED

    def test_predicts_most_frequent_class(self, xy):
        X, y = xy
        pred = model_dummy.build_model().fit(X, y).predict(X)
        assert (pred == np.bincount(y).argmax()).all()

    def test_has_no_tunable_params(self):
        with pytest.raises(TypeError):
            model_dummy.build_model(strategy="prior")
