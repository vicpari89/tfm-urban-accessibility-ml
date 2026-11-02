"""Evaluation metrics.

Metrics chosen in the thesis (3.3.3), all insensitive to the majority class:
    - macro F1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html, with param average: 'macro'
    - balanced accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    - quadratic weighted kappa: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html, with param weights: 'quadratic'
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

from urban_accessibility_ml.config import CLASS_ORDER, METRICS

KAPPA_LABELS = list(range(len(CLASS_ORDER))) # Used in case one fold does not have class A (or any other class).

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 averaged over classes with the same weight each."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall per class."""
    return balanced_accuracy_score(y_true, y_pred)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa."""
    return cohen_kappa_score(y_true, y_pred, labels=KAPPA_LABELS, weights="quadratic")


# Metric function constants
METRIC_FUNCTIONS = {
    "f1_macro": macro_f1,
    "balanced_accuracy": balanced_accuracy,
    "quadratic_weighted_kappa": quadratic_weighted_kappa,
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Metrics of one experiment on one single fold.
    :param y_true: real classes, encoded (C=0, B=1, A=2).
    :param y_pred: predicted classes, same encoding.
    :return: {metric name: value}, one key per METRICS.
    """
    return {name: METRIC_FUNCTIONS[name](y_true, y_pred) for name in METRICS}


def summarize_folds(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Mean and std of each metric across folds.
    :param results: one row per experiment and fold.
    :param group_cols: columns that identify an experiment (its folds are grouped).
    :return: one row per group, with "<metric>_mean" and "<metric>_std" columns.
    """
    summary = results.groupby(group_cols)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()
