import pandas as pd
import numpy as np
from typing import Iterator, Tuple, List, Dict, Any
from src.utils.logging_utils import get_logger

logger = get_logger("expanding_window")

class ExpandingWindowSplitter:
    """
    Expanding-window out-of-sample splitter for time series.
    
    Leakage control: with horizon h,
        max(train_date) <= min(test_date) - h months
    
    This ensures that target windows of the training set do not overlap
    with the test period.
    """
    def __init__(self, horizon: int, test_size_months: int, 
                 initial_train_end: str, gap_equals_horizon: bool = True, gap_months: int = None):
        self.horizon = horizon
        self.test_size_months = test_size_months
        self.initial_train_end = pd.to_datetime(initial_train_end)
        self.gap_equals_horizon = gap_equals_horizon
        self.gap_months = gap_months
        self.diagnostics: List[Dict[str, Any]] = []
        
    def split(self, df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices for training and testing splits.
        df must have a 'DATE' column (sorted chronologically).
        """
        df_sorted = df.sort_values('DATE').reset_index(drop=True)
        dates = pd.to_datetime(df_sorted['DATE'])
        min_date = dates.min()
        max_date = dates.max()
        
        # Test period starts 1 month after initial_train_end
        first_test_start = self.initial_train_end + pd.DateOffset(months=1)
        
        current_test_start = first_test_start
        split_id = 1
        self.diagnostics = []
        
        while True:
            current_test_end = current_test_start + pd.DateOffset(months=self.test_size_months - 1)
            
            if current_test_start > max_date:
                break
                
            if current_test_end > max_date:
                current_test_end = max_date
                
            gap_val = self.gap_months if self.gap_months is not None else (self.horizon if self.gap_equals_horizon else 0)
            current_train_end = current_test_start - pd.DateOffset(months=gap_val)
            
            # Select indices based on sorted DataFrame
            train_mask = (dates >= min_date) & (dates <= current_train_end)
            train_idx = df_sorted[train_mask].index.values
            
            test_mask = (dates >= current_test_start) & (dates <= current_test_end)
            test_idx = df_sorted[test_mask].index.values
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                break
                
            # Get target statistics
            target_cols = [c for c in df_sorted.columns if 'target' in c.lower() or 'recession' in c.lower()]
            pos_train = 0
            pos_test = 0
            if target_cols:
                # Find the target column corresponding to this horizon if available, otherwise first match
                matching_col = [c for c in target_cols if f"_{self.horizon}" in c]
                t_col = matching_col[0] if matching_col else target_cols[0]
                pos_train = int(df_sorted.loc[train_idx, t_col].dropna().sum())
                pos_test = int(df_sorted.loc[test_idx, t_col].dropna().sum())
                
            self.diagnostics.append({
                "horizon": self.horizon,
                "split_id": split_id,
                "train_start": min_date.strftime("%Y-%m-%d"),
                "train_end_raw": (current_test_start - pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                "train_end_after_gap": current_train_end.strftime("%Y-%m-%d"),
                "test_start": current_test_start.strftime("%Y-%m-%d"),
                "test_end": current_test_end.strftime("%Y-%m-%d"),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "positive_train": pos_train,
                "positive_test": pos_test
            })
            
            yield train_idx, test_idx
            
            if current_test_end >= max_date:
                break
                
            current_test_start = current_test_start + pd.DateOffset(months=self.test_size_months)
            split_id += 1
            
    def get_split_diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame(self.diagnostics)
