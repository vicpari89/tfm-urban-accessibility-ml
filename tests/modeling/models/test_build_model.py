import numpy as np
import pytest

from urban_accessibility_ml.config import SEED
from urban_accessibility_ml.modeling.models import BUILDERS, build_model

MODEL_NAMES = ["dummy", "logistic_regression"]
OTHER_SEED = 7
UNKNOWN_MODEL = "does_not_exist"
CLASSES = {0, 1, 2}  # C, B, A


class TestBuildModel:
    def test_builders_names(self):
        assert list(BUILDERS) == MODEL_NAMES

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
