import numpy as np
import pandas as pd
from typing import List, Dict, Any

def compute_lead_time(dates: pd.Series, y_true: np.ndarray, y_prob: np.ndarray, 
                      threshold: float, recession_episodes: List[Dict[str, Any]], 
                      lead_up_months: int = 12) -> List[Dict[str, Any]]:
    """
    Calculate lead-time statistics for each recession episode.
    
    For each episode:
    - detected: True if predicted prob >= threshold in the lead_up_months prior to start
    - lead_time_months: months between first warning in lead-up window and recession start
    - peak_probability: max predicted prob in lead-up + recession window
    - false_warning_months: number of months where prob >= threshold but not inside any lead-up/recession window
    """
    dates = pd.to_datetime(dates)
    results = []
    
    is_recession = np.zeros(len(dates), dtype=bool)
    is_lead_up = np.zeros(len(dates), dtype=bool)
    
    episode_info = []
    for ep in recession_episodes:
        ep_start = pd.to_datetime(ep['start'])
        ep_end = pd.to_datetime(ep['end'])
        
        lu_start = ep_start - pd.DateOffset(months=lead_up_months)
        lu_end = ep_start - pd.DateOffset(months=1)
        
        ep_mask = (dates >= ep_start) & (dates <= ep_end)
        lu_mask = (dates >= lu_start) & (dates <= lu_end)
        
        is_recession |= ep_mask.values
        is_lead_up |= lu_mask.values
        
        episode_info.append({
            "name": ep['name'],
            "start": ep_start,
            "end": ep_end,
            "lu_start": lu_start,
            "lu_end": lu_end,
            "ep_mask": ep_mask.values,
            "lu_mask": lu_mask.values
        })
        
    # Warnings at threshold
    warnings = y_prob >= threshold
    false_warnings = warnings & (~is_recession) & (~is_lead_up)
    total_false_warning_months = int(np.sum(false_warnings))
    
    for ep in episode_info:
        # Check warnings in lead-up window
        lu_warnings = warnings & ep['lu_mask']
        detected = bool(np.any(lu_warnings))
        
        lead_time = 0
        if detected:
            warning_indices = np.where(lu_warnings)[0]
            first_warning_date = dates.iloc[warning_indices[0]]
            lead_time = (ep['start'].year - first_warning_date.year) * 12 + (ep['start'].month - first_warning_date.month)
            
        # Peak prob in lead-up + recession
        window_mask = ep['lu_mask'] | ep['ep_mask']
        peak_prob = float(np.max(y_prob[window_mask])) if np.any(window_mask) else 0.0
        
        results.append({
            "episode": ep['name'],
            "detected": detected,
            "lead_time_months": lead_time,
            "peak_probability": peak_prob,
            "false_warning_months": total_false_warning_months
        })
        
    return results
