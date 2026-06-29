import pandas as pd
import numpy as np
from typing import List, Dict, Any

def track_feature_stability(importance_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Consolidate feature importances across splits/years to track feature stability.
    Each dictionary in the list should contain:
    - "horizon": int
    - "split_id": int
    - "feature": str
    - "importance": float
    """
    if not importance_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(importance_list)
    return df
