"""Feature sets and target for the models.

Decides which columns are used as input, given a feature set (origin of the data) and a profile.

Inputs are the scores per dimension and profile (puntuacion_<dimension>_<profile>).
The four total scores (puntuacion_total*) are never used.
"""

import pandas as pd

from urban_accessibility_ml.config import (
    CLASS_ORDER,
    DIMENSIONS,
    FEATURE_SETS,
    PROFILES,
    TARGET_COLUMNS,
    THRESHOLD,
    VIABILITY,
)


def dimensions_for(feature_set: str) -> list[str]:
    """
    Dimensions usable as inputs in a feature set. A dimension is viable from a source when
    VIABILITY[dimension][source] >= THRESHOLD.
    - cartografico_aereo: viable from the cartography or the aerial imagery.
    - calle: viable from the street-level imagery.
    - combinado: viable from any of the three sources.
    - completo: all the dimensions.
    """
    sources = {
        "cartografico_aereo": ["cartografia", "aerea"],
        "calle": ["calle"],
        "combinado": ["cartografia", "aerea", "calle"],
    }
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if feature_set == "completo":
        return list(DIMENSIONS)
    return [d for d in DIMENSIONS if max(VIABILITY[d][s] for s in sources[feature_set]) >= THRESHOLD]


def encode_class(classes: pd.Series) -> pd.Series:
    """Classes as ordered integers following CLASS_ORDER: C -> 0, B -> 1, A -> 2."""
    codes = classes.map({c: i for i, c in enumerate(CLASS_ORDER)})
    if codes.isna().any():
        raise ValueError(f"Unknown classes: {sorted(classes[codes.isna()].astype(str).unique())}")
    return codes.astype(int)


def check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if a column is missing from the dataframe or is a total score."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not in the dataframe: {missing}")
    leaks = [c for c in columns if c.startswith("puntuacion_total")]
    if leaks:
        raise ValueError(f"Total scores cannot be used as inputs (target leakage): {leaks}")


def get_xy(df: pd.DataFrame, feature_set: str, profile: str, include_all_profiles: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Inputs X and target y for a feature set and a profile.
    :param df: Dataframe.
    :param feature_set: Feature set with values "cartografico_aereo", "calle", "combinado", "completo".
    :param profile: one of the profiles: "global", "funcional", "sensorial", "cognitiva".
    :param include_all_profiles: Include or not all the profiles. False by default.
    :return: inputs set for X, class for Y.
    """

    # 1. Validate the arguments
    if feature_set not in FEATURE_SETS:  raise ValueError
    if profile not in PROFILES:          raise ValueError

    # 2. Get the dimensions of this feature set
    dimensions = dimensions_for(feature_set)

    # 3. Get column names
    if include_all_profiles:
        columns = [f"puntuacion_{d}_{p}" for d in dimensions for p in PROFILES]
    else: # Just for one profile
        columns = [f"puntuacion_{d}_{profile}" for d in dimensions]

    # 4. Safety checks
    check_columns(df, columns)

    # 5. Inputs X: only those columns, keeping the index (key_tramo)
    X = df[columns]

    # 6. Target y: class of the profile, as ordered integers
    target = TARGET_COLUMNS[profile]
    y = encode_class(df[target])  # C -> 0, B -> 1, A -> 2

    # 7. Return both, with the same index
    return X, y
