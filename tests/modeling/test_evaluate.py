import numpy as np
import pandas as pd
import pytest

from urban_accessibility_ml.config import METRICS
from urban_accessibility_ml.modeling.evaluate import (
    METRIC_FUNCTIONS,
    balanced_accuracy,
    evaluate,
    macro_f1,
    quadratic_weighted_kappa,
    summarize_folds,
)

# Confusion matrix (rows = true C, B, A; columns = predicted):
#   C: [1, 1, 0]   B: [0, 2, 0]   A: [1, 0, 1]
Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])

Y = np.array([0, 0, 0, 0, 1, 1, 2, 2])  # imbalanced, like the real data


# --- individual metrics ------------------------------------------------------------------------


def test_macro_f1_hand_computed():
    # F1 per class: 0.5, 0.8, 2/3
    assert macro_f1(Y_TRUE, Y_PRED) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_balanced_accuracy_hand_computed():
    # recall per class: 0.5, 1, 0.5
    assert balanced_accuracy(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)


def test_qwk_hand_computed():
    # 1 - observed / expected = 1 - 5 / 7
    assert quadratic_weighted_kappa(Y_TRUE, Y_PRED) == pytest.approx(2 / 7)


@pytest.mark.parametrize("metric", [macro_f1, balanced_accuracy, quadratic_weighted_kappa])
def test_metric_perfect_prediction(metric):
    assert metric(Y, Y) == 1.0


@pytest.mark.parametrize("metric", [macro_f1, balanced_accuracy])
def test_metric_all_wrong_is_zero(metric):
    assert metric(Y_TRUE, (Y_TRUE + 1) % 3) == 0.0


def test_qwk_all_wrong_is_negative():
    assert quadratic_weighted_kappa(Y_TRUE, (Y_TRUE + 1) % 3) < 0


def test_always_majority_class_scores_low():
    pred = np.zeros_like(Y)
    assert balanced_accuracy(Y, pred) == pytest.approx(1 / 3)  # recalls 1, 0, 0
    assert macro_f1(Y, pred) == pytest.approx(0.2222, abs=1e-3)  # only class C has F1 (2/3)
    assert quadratic_weighted_kappa(Y, pred) == 0.0  # no agreement beyond chance


def test_qwk_penalises_distant_errors_more():
    y = np.array([0, 1, 2, 0, 1, 2])
    near = np.array([1, 1, 2, 0, 1, 1])  # C->B, A->B
    far = np.array([2, 1, 2, 0, 1, 0])  # C->A, A->C
    assert quadratic_weighted_kappa(y, far) < quadratic_weighted_kappa(y, near)


def test_metrics_accept_lists_and_series():
    assert macro_f1(list(Y_TRUE), list(Y_PRED)) == macro_f1(pd.Series(Y_TRUE), pd.Series(Y_PRED))


# --- fold without class A ----------------------------------------------------------------------


@pytest.mark.parametrize("metric", [macro_f1, balanced_accuracy, quadratic_weighted_kappa])
def test_fold_without_class_a_perfect_prediction_is_one(metric):
    y = np.array([0, 0, 1, 1])
    assert metric(y, y) == 1.0


def test_fold_without_class_a_but_predicted_a_is_penalised():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 2])
    assert macro_f1(y, pred) < 1.0
    assert quadratic_weighted_kappa(y, pred) < 1.0


# --- evaluate ----------------------------------------------------------------------------------


def test_metric_functions_cover_metrics():
    assert set(METRIC_FUNCTIONS) == set(METRICS)


def test_evaluate_keys_follow_metrics():
    assert list(evaluate(Y_TRUE, Y_PRED)) == METRICS


def test_evaluate_matches_individual_metrics():
    out = evaluate(Y_TRUE, Y_PRED)
    for name in METRICS:
        assert out[name] == METRIC_FUNCTIONS[name](Y_TRUE, Y_PRED)


def test_evaluate_perfect_prediction():
    assert evaluate(Y, Y) == {m: 1.0 for m in METRICS}


def test_evaluate_returns_plain_floats():
    assert all(isinstance(v, float) for v in evaluate(Y_TRUE, Y_PRED).values())


# --- summarize_folds ---------------------------------------------------------------------------


def _results(rows):
    """rows: (model, profile, fold, value); every metric gets the same value."""
    return pd.DataFrame(
        [{"model": m, "profile": p, "fold": f, **{k: v for k in METRICS}} for m, p, f, v in rows]
    )


def test_summarize_folds_mean_and_std():
    df = _results([("dummy", "g", 0, 0.2), ("dummy", "g", 1, 0.4), ("logreg", "g", 0, 0.6), ("logreg", "g", 1, 0.8)])
    out = summarize_folds(df, ["model"])
    assert out["model"].tolist() == ["dummy", "logreg"]
    assert out["f1_macro_mean"].tolist() == pytest.approx([0.3, 0.7])
    assert out["f1_macro_std"].tolist() == pytest.approx([0.1414, 0.1414], abs=1e-3)  # sample std (n-1)


def test_summarize_folds_columns():
    out = summarize_folds(_results([("dummy", "g", 0, 0.5)]), ["model", "profile"])
    expected = ["model", "profile"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    assert out.columns.tolist() == expected  # fold column is not carried over


def test_summarize_folds_groups_by_all_group_cols():
    df = _results(
        [("m", "global", 0, 0.2), ("m", "global", 1, 0.4), ("m", "funcional", 0, 0.8), ("m", "funcional", 1, 1.0)]
    )
    out = summarize_folds(df, ["model", "profile"]).set_index("profile")
    assert len(out) == 2
    assert out.loc["global", "quadratic_weighted_kappa_mean"] == pytest.approx(0.3)
    assert out.loc["funcional", "balanced_accuracy_mean"] == pytest.approx(0.9)


def test_summarize_folds_single_fold_has_nan_std():
    out = summarize_folds(_results([("m", "g", 0, 0.5)]), ["model"])
    assert out["f1_macro_mean"].iloc[0] == 0.5
    assert np.isnan(out["f1_macro_std"].iloc[0])


def test_summarize_folds_ignores_non_metric_columns():
    df = _results([("m", "g", 0, 0.5), ("m", "g", 1, 0.5)]).assign(n_train=100)
    assert "n_train_mean" not in summarize_folds(df, ["model"]).columns


def test_summarize_folds_does_not_modify_input():
    df = _results([("m", "g", 0, 0.5), ("m", "g", 1, 0.7)])
    before = df.copy()
    summarize_folds(df, ["model"])
    pd.testing.assert_frame_equal(df, before)
