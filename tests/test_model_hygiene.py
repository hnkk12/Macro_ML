import pandas as pd
import numpy as np
import pytest
from src.models.baselines import StatsmodelWrapper
from src.models.ensembles import HybridProbitMLEnsemble

def test_statsmodel_wrapper_alignment_hygiene():
    # Mismatched indexes: X has reset index, y has non-reset index
    y = pd.Series([1, 0, 1, 0, 1], index=[10, 11, 12, 13, 14])
    X = pd.DataFrame({'feat': [0.1, -0.5, 1.2, -0.2, 0.8]}, index=[0, 1, 2, 3, 4])
    
    # Instantiate Probit wrapper
    model = StatsmodelWrapper(model_type="probit")
    
    # If the bug is fixed, fitting should succeed without falling back to sklearn
    model.fit(X, y)
    
    # If it fell back, fallback_model would be not None
    assert model.fallback_model is None, "StatsmodelWrapper fell back to scikit-learn when it should have aligned indices!"
    assert model.results is not None

def test_ensemble_intercept_hygiene():
    ensemble = HybridProbitMLEnsemble(random_state=42)
    # The meta-learner MUST have fit_intercept=True
    assert ensemble.meta_learner.fit_intercept is True, "Meta-learner should fit intercept to avoid flat 0.5 probabilities!"
