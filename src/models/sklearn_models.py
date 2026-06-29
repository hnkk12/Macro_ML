from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_logistic_l2(C: float = 1.0, solver: str = 'lbfgs', max_iter: int = 1000, random_state: int = 42):
    return LogisticRegression(C=C, penalty='l2', solver=solver, max_iter=max_iter, random_state=random_state)

def get_random_forest(n_estimators: int = 100, max_depth: int = None, min_samples_split: int = 2, random_state: int = 42):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
