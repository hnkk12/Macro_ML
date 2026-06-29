import pandas as pd
import numpy as np
import pytest
from src.features.feature_sets import compute_features
from src.features.target_builder import make_forward_target

def test_feature_target_alignment():
    # Make a dummy panel
    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    df = pd.DataFrame({
        'DATE': dates,
        'PAYEMS': np.linspace(100, 110, 24),
        'UNRATE': np.linspace(5, 4, 24),
        'FEDFUNDS': np.linspace(2, 1, 24),
        'CPIAUCSL': np.linspace(200, 210, 24),
        'GS10': np.linspace(3, 2, 24),
        'GS5': np.linspace(2.5, 1.8, 24),
        'TB3MS': np.linspace(1.5, 0.8, 24),
        'INDPRO': np.linspace(100, 105, 24),
        'T10Y2Y': np.linspace(0.5, 0.2, 24),
        'BAMLH0A0HYM2': np.linspace(1.2, 1.5, 24),
        '^GSPC': np.linspace(3000, 3200, 24),
        'USREC': np.zeros(24)
    })
    
    # Compute features
    feats = compute_features(df)
    assert len(feats) == len(df)
    assert (feats['DATE'] == df['DATE']).all()
    
    # Compute target for horizon 3
    target = make_forward_target(df['USREC'], horizon=3)
    assert len(target) == len(df)
    assert target.index.equals(df.index)
