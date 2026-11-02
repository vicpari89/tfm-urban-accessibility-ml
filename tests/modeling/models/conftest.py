import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

N_ROWS_PER_CLASS = {0: 40, 1: 30, 2: 10}  # C, B, A: imbalanced like the real data
N_FEATURES = 4
DATA_SEED = 0


@pytest.fixture
def xy():
    """Small imbalanced problem with the 3 encoded classes and separable features."""
    rng = np.random.default_rng(DATA_SEED)
    y = np.concatenate([np.full(n, c) for c, n in N_ROWS_PER_CLASS.items()])
    X = rng.normal(size=(len(y), N_FEATURES)) + y[:, None]
    return X, y


@pytest.fixture
def dummy_accuracy(xy):
    """Accuracy of the dummy on the fixture: the floor any real model must beat."""
    X, y = xy
    return (DummyClassifier(strategy="most_frequent").fit(X, y).predict(X) == y).mean()
