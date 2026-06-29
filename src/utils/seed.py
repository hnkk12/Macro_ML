import random
import numpy as np

def set_seed(seed: int = 42):
    """Set global seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Note: xgboost/lightgbm/scikit-learn seeds are usually set inside model parameters.
