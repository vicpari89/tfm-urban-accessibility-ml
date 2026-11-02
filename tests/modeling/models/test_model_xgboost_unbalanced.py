from xgboost import XGBClassifier

from urban_accessibility_ml.modeling.models import model_xgboost_balanced, model_xgboost_unbalanced

OTHER_SEED = 7


class TestXGBoostUnbalanced:
    def test_type_and_params_are_the_same_as_xgboost_balanced(self):
        model = model_xgboost_unbalanced.build_model(OTHER_SEED)
        assert isinstance(model, XGBClassifier)
        assert model.get_params() == model_xgboost_balanced.build_model(OTHER_SEED).get_params()

    def test_does_not_need_sample_weight(self):
        assert model_xgboost_unbalanced.NEED_SAMPLE_WEIGHT is False

    def test_beats_the_dummy(self, xy, dummy_accuracy):
        X, y = xy
        pred = model_xgboost_unbalanced.build_model().fit(X, y).predict(X)
        assert (pred == y).mean() > dummy_accuracy
