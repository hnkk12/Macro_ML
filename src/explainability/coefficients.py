import numpy as np
import pandas as pd

def extract_standardized_coefficients(model, feature_names, std_x: np.ndarray, horizon: int, split_id: int) -> pd.DataFrame:
    """
    Extract standardized coefficients for a linear/logistic model.
    coef_standardized = coef_raw * std_x
    """
    # Extract raw coefficients depending on model type (sklearn vs statsmodels)
    if hasattr(model, "coef_"):
        # For sklearn models, coef_ is shape (1, n_features) or (n_features,)
        coef_raw = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
    elif hasattr(model, "fallback_model") and model.fallback_model is not None:
        coef_raw = model.fallback_model.coef_[0] if len(model.fallback_model.coef_.shape) > 1 else model.fallback_model.coef_
    elif hasattr(model, "results") and model.results is not None:
        # For statsmodels
        params = model.results.params
        coef_raw = []
        for col in feature_names:
            if col in params:
                coef_raw.append(params[col])
            else:
                coef_raw.append(0.0)
        coef_raw = np.array(coef_raw)
    else:
        # Model does not have coefficients (e.g. Random Forest, XGBoost)
        return pd.DataFrame()
        
    # Standardized coefficients = raw_coef * std_X
    # (Since standardizing features divides by std, the coef must be multiplied by std to be comparable to standard deviation units)
    std_coef = coef_raw * std_x
    
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef_raw,
        "standardized_coefficient": std_coef,
        "horizon": horizon,
        "split_id": split_id
    })
    return df
