"""Search spaces and settings of the hyperparameter tuning (see tuning_search.py).

Every key of a "space" must be a keyword argument of the build_model of that model
(modeling/models/model_*.py).
"""

from scipy.stats import loguniform, randint, uniform

from urban_accessibility_ml.config import FEATURE_SETS

TUNING_METRIC = "f1_macro"  # one of config.METRICS
TUNING_INNER_SPLITS = 3  # inner folds (StratifiedGroupKFold) inside the train rows of a partition
TUNING_N_CANDIDATES = 30  # candidates of a "random" search; a "grid" tries all of them
TUNING_TIE_TOLERANCE = 0.005  # candidates this close to the best are a tie: the simplest wins
TUNING_FEATURE_SET = "completo"  # reference feature set and profile of the tuning
TUNING_PROFILE = "global"
# Order of the tuning: the reference set first, then the others
TUNING_FEATURE_SET_ORDER = [TUNING_FEATURE_SET] + [
    f for f in FEATURE_SETS if f != TUNING_FEATURE_SET
]

C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# Per model: "grid" (every combination of the lists) or "random" (a sample of the space).
SEARCH_SPACES = {
    "logistic_regression": {
        "kind": "grid",
        # A list of grids: each penalty goes with a solver that supports it (lbfgs does not do l1)
        "space": [
            {"C": C_VALUES, "penalty": ["l2"], "solver": ["lbfgs"]},
            {"C": C_VALUES, "penalty": ["l1"], "solver": ["saga"]},
        ],
    },
    "random_forest": {
        "kind": "random",
        "space": {
            "max_depth": randint(3, 16),  # 3-15
            "min_samples_leaf": randint(1, 11),  # 1-10
            "max_features": uniform(0.3, 0.7),  # 0.3-1.0
        },
    },
    "lightgbm": {
        "kind": "random",
        "space": {
            "learning_rate": loguniform(0.01, 0.3),
            "num_leaves": randint(7, 64),  # 7-63
            "n_estimators": randint(100, 501),  # 100-500
            "min_child_samples": randint(5, 21),  # 5-20
            "subsample": uniform(0.6, 0.4),  # 0.6-1.0
            "colsample_bytree": uniform(0.6, 0.4),  # 0.6-1.0
        },
    },
    "xgboost_balanced": {
        "kind": "random",
        "space": {
            "learning_rate": loguniform(0.01, 0.3),
            "max_depth": randint(2, 7),  # 2-6
            "n_estimators": randint(100, 501),  # 100-500
            "subsample": uniform(0.6, 0.4),  # 0.6-1.0
            "colsample_bytree": uniform(0.6, 0.4),  # 0.6-1.0
            "min_child_weight": randint(1, 11),  # 1-10
        },
    },
}
