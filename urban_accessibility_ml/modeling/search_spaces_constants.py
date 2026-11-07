"""Search spaces and settings of the hyperparameter tuning (see tuning.py).

Every key of a "space" must be a keyword argument of the build_model of that model
(modeling/models/model_*.py).
"""

TUNING_METRIC = "f1_macro"  # one of config.METRICS
TUNING_INNER_SPLITS = 3  # inner folds (StratifiedGroupKFold) inside the train rows of a partition
TUNING_TIE_TOLERANCE = 0.005  # candidates this close to the best are a tie: the simplest wins
TUNING_FEATURE_SET = "completo"  # reference feature set and profile of the tuning
TUNING_PROFILE = "global"

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
}
