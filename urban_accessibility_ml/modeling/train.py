"""Model training (pending)."""

import pandas as pd

from urban_accessibility_ml.preprocessing.split import Split


def run_one(
    df: pd.DataFrame,
    split: Split,
    feature_set: str,
    profile: str,
    model_name: str,
) -> dict:
    """Train and evaluate ONE experiment on ONE split."""
    raise NotImplementedError
