"""Train/test splits: grouped by street, stratified by class, reproducible with a fixed seed.

Strategies:
    - within_city: grouped inside one city (Albacete or Algeciras).
    - cross_city: train on one city, test on the other city (and viceversa).
    - mixed: train mixing both cities, and test both too.

In this file, just the splitting.

We use StratifiedGroupKFold because it combines StratifiedKFold and GroupKFold:
    - StratifiedKFold does not check the streets, so the same street could be in test and train.
    - GroupKFold: check the street tramos are only on test or train, but does not check the classes are balanced in train/test.
    - Ref: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_SPLITS = 5

CITY_COL = "ciudad"
STREET_COL = "nombre_calle_base"
BARRIO_COL = "barrio"
ID_COL = "key_tramo"  # ciudad + id_tramo
CLASS_COL = "clase_total"


@dataclass(frozen=True)
class Split:
    """One train/test partition, stored as row identifiers (not as copies of the data)."""

    strategy: str
    name: str
    fold: int
    train_ids: np.ndarray
    test_ids: np.ndarray

def split_within_city(df: pd.DataFrame, city: str, group_col: str = STREET_COL):
    # 1. Filter rows of the city
    city_df = df[df[CITY_COL] == city]

    # 2. Create the groups for each row
    groups = city_df[group_col]

    # 3. Stratifier
    y = city_df[CLASS_COL]

    # 4. Create the splitter
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # 5. Build one Split per fold
    return [
        Split("within_city", city, fold, city_df[ID_COL].to_numpy()[tr], city_df[ID_COL].to_numpy()[te])
        for fold, (tr, te) in enumerate(splitter.split(city_df, y, groups))
    ]

def split_cross_city(df: pd.DataFrame, train_city: str, test_city: str):
    # 1. Filter by train_city for the train_df
    train_df = df[df[CITY_COL] == train_city]

    # 2. Filter by test_city for the test_df
    test_df = df[df[CITY_COL] == test_city]

    # 3. Single split (no folds): all of train_city -> all of test_city
    return [Split("cross_city", f"{train_city}->{test_city}", 0, train_df[ID_COL].to_numpy(), test_df[ID_COL].to_numpy())]

def split_mixed(df: pd.DataFrame, group_col: str = STREET_COL):
    # 1.Crete a group formed by CITY_COL and STREET_COL
    groups = df[CITY_COL] + "_" + df[group_col].astype(str)

    # 2. Stratifier class
    y = df[CLASS_COL]

    # 3. Create the splitter
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # 4. Build one Split per fold
    ids = df[ID_COL].to_numpy()
    return [
        Split("mixed", "both", fold, ids[tr], ids[te])
        for fold, (tr, te) in enumerate(splitter.split(df, y, groups))
    ]
