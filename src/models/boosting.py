from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_xgboost(max_depth: int = 3, min_child_weight: float = 1.0, reg_lambda: float = 1.0, 
                learning_rate: float = 0.1, n_estimators: int = 100, random_state: int = 42):
    # xgboost has verbosity/silent parameters. Verbosity=0 is silent.
    return XGBClassifier(
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        verbosity=0,
        random_state=random_state,
        eval_metric='logloss'
    )

def get_lightgbm(max_depth: int = 3, num_leaves: int = 31, reg_lambda: float = 0.0, 
                 learning_rate: float = 0.1, n_estimators: int = 100, random_state: int = 42):
    return LGBMClassifier(
        max_depth=max_depth,
        num_leaves=num_leaves,
        reg_lambda=reg_lambda,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        verbosity=-1,
        random_state=random_state
    )
