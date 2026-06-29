import numpy as np
from scipy.stats import norm
from src.utils.logging_utils import get_logger

logger = get_logger("significance")

def diebold_mariano_test(y_true: np.ndarray, y_prob1: np.ndarray, y_prob2: np.ndarray, horizon: int = 1) -> dict:
    """
    Perform the Diebold-Mariano test to compare the forecast accuracy of two models.
    
    Using Brier score (squared error) as the loss function:
    L(y, p) = (y - p)^2
    
    Parameters:
    -----------
    y_true: np.ndarray
        Ground truth labels (0 or 1)
    y_prob1: np.ndarray
        Forecast probabilities from model 1
    y_prob2: np.ndarray
        Forecast probabilities from model 2 (usually the main proposed model)
    horizon: int
        Forecast horizon. Autocorrelation is corrected up to lag (horizon - 1).
        
    Returns:
    --------
    dict
        A dictionary containing the DM statistic, p-value, and mean loss differential.
    """
    y_true = np.array(y_true, dtype=float)
    y_prob1 = np.array(y_prob1, dtype=float)
    y_prob2 = np.array(y_prob2, dtype=float)
    
    n = len(y_true)
    if n <= 1:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_diff": np.nan}
        
    # Calculate loss differential d_t = Loss(Model 1) - Loss(Model 2)
    # A positive d_t means Model 2 has smaller loss (better accuracy)
    loss1 = (y_true - y_prob1) ** 2
    loss2 = (y_true - y_prob2) ** 2
    d = loss1 - loss2
    
    mean_d = np.mean(d)
    
    # Compute autocovariances up to lag h-1
    # If horizon is h, forecast errors can be autocorrelated up to h-1 lags
    max_lag = max(1, horizon - 1)
    
    # Variance of the mean loss differential (with Newey-West type variance estimator)
    # Gamma_0 (variance at lag 0)
    gamma = np.zeros(max_lag + 1)
    d_demeaned = d - mean_d
    gamma[0] = np.mean(d_demeaned ** 2)
    
    # Gamma_k (covariance at lag k)
    for k in range(1, max_lag + 1):
        if n - k > 0:
            gamma[k] = np.mean(d_demeaned[k:] * d_demeaned[:-k])
        else:
            gamma[k] = 0.0
            
    # Standard error of mean_d
    # V(mean_d) = (1/n) * [gamma_0 + 2 * sum_{k=1}^{h-1} (1 - k/h) * gamma_k] (Bartlett kernel / Newey-West weight)
    var_d = gamma[0]
    for k in range(1, max_lag + 1):
        weight = 1.0 - (k / (max_lag + 1))
        var_d += 2 * weight * gamma[k]
        
    # Standard error of mean_d
    se_d = np.sqrt(max(1e-15, var_d) / n)
    
    if se_d <= 1e-12:
        dm_stat = 0.0
    else:
        dm_stat = mean_d / se_d
        
    # Two-sided p-value from normal distribution
    # H0: E(d_t) = 0
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))
    
    # One-sided p-value (H1: E(d_t) > 0, i.e., Model 2 is significantly better than Model 1)
    p_value_one_sided = 1.0 - norm.cdf(dm_stat)
    
    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "p_value_one_sided": float(p_value_one_sided),
        "mean_diff": float(mean_d)
    }
