import pandas as pd
import numpy as np
import pytest
from src.validation.expanding_window import ExpandingWindowSplitter

def test_temporal_gap_control():
    # Make dummy dataframe with dates from 2000-01-01 to 2010-12-01 (132 months)
    dates = pd.date_range('2000-01-01', periods=132, freq='MS')
    df = pd.DataFrame({
        'DATE': dates,
        'val': np.arange(132),
        'Recession_within_6mo': np.zeros(132)
    })
    
    horizon = 6
    splitter = ExpandingWindowSplitter(
        horizon=horizon,
        test_size_months=12,
        initial_train_end='2004-12-01',
        gap_equals_horizon=True
    )
    
    splits = list(splitter.split(df))
    assert len(splits) > 0
    
    diagnostics = splitter.get_split_diagnostics()
    
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        diag = diagnostics.iloc[split_idx]
        test_start_val = pd.to_datetime(diag['test_start'])
        train_end_val = pd.to_datetime(diag['train_end_after_gap'])
        
        # Test temporal gap constraint: train_end_after_gap <= test_start - horizon months
        expected_train_end = test_start_val - pd.DateOffset(months=horizon)
        assert train_end_val <= expected_train_end
        
        train_dates = pd.to_datetime(df.loc[train_idx, 'DATE'])
        test_dates = pd.to_datetime(df.loc[test_idx, 'DATE'])
        
        assert train_dates.max() <= expected_train_end
        assert test_dates.min() >= test_start_val
        
        # Train index of split N should expand (be a superset of split N-1)
        if split_idx > 0:
            prev_train_idx = splits[split_idx-1][0]
            assert set(prev_train_idx).issubset(set(train_idx))
