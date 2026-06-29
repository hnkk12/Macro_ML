import numpy as np
from typing import Callable, Tuple

def block_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn: Callable[[np.ndarray, np.ndarray], float], 
                       block_size: int = 6, n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Block bootstrap to calculate confidence intervals for a metric.
    block_size = 6 months to preserve temporal autocorrelation.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    
    if n == 0:
        return np.nan, np.nan
        
    num_blocks = int(np.ceil(n / block_size))
    boot_metrics = []
    
    max_start_idx = n - block_size
    
    for _ in range(n_boot):
        if max_start_idx <= 0:
            # Fallback to standard bootstrap if too short
            indices = np.random.choice(n, size=n, replace=True)
        else:
            # Draw block start indices
            start_indices = np.random.choice(max_start_idx + 1, size=num_blocks, replace=True)
            indices = []
            for start in start_indices:
                indices.extend(range(start, min(start + block_size, n)))
            indices = np.array(indices[:n])
            
            # Fill up to exactly n if needed
            if len(indices) < n:
                extra = np.random.choice(n, size=(n - len(indices)), replace=True)
                indices = np.concatenate([indices, extra])
                
        try:
            score = metric_fn(y_true[indices], y_prob[indices])
            if not np.isnan(score):
                boot_metrics.append(score)
        except Exception:
            pass
            
    if len(boot_metrics) == 0:
        return np.nan, np.nan
        
    alpha = 1.0 - ci
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)
    
    lower_ci = np.percentile(boot_metrics, lower_pct)
    upper_ci = np.percentile(boot_metrics, upper_pct)
    
    return float(lower_ci), float(upper_ci)
