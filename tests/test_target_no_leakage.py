import pandas as pd
import numpy as np
import pytest
from src.features.target_builder import make_forward_target

def test_target_no_current_leakage():
    # Test 1: target at month T should not use usrec[T]
    usrec = pd.Series([1, 0, 0, 0], index=pd.date_range('2020-01-01', periods=4, freq='MS'))
    # horizon = 2: at t=0 (2020-01-01), usrec[0] is 1, but lookahead is usrec[1]=0, usrec[2]=0.
    # Target should be 0, not affected by usrec[0] being 1.
    y = make_forward_target(usrec, horizon=2)
    assert y.iloc[0] == 0

def test_end_of_sample_nan():
    # Test 2: end of sample must have NaN when not enough horizon
    usrec = pd.Series([0, 0, 0, 0, 0], index=pd.date_range('2020-01-01', periods=5, freq='MS'))
    y = make_forward_target(usrec, horizon=3)
    # last 3 observations should be NaN because we need t+1, t+2, t+3
    # t=4 (last one): need t=5, t=6, t=7 (all NaN)
    # t=3: need t=4, t=5, t=6 (NaN)
    # t=2: need t=3, t=4, t=5 (NaN)
    # t=1: need t=2, t=3, t=4 -> 0, 0, 0 -> 0
    assert np.isnan(y.iloc[-1])
    assert np.isnan(y.iloc[-2])
    assert np.isnan(y.iloc[-3])
    assert y.iloc[-4] == 0

def test_horizon_window():
    # Test 3: with horizon=3, window [t+1, t+3] is correct
    usrec = pd.Series([0, 0, 0, 1, 0, 0], index=pd.date_range('2020-01-01', periods=6, freq='MS'))
    # at t=0: lookahead is usrec[1]=0, usrec[2]=0, usrec[3]=1. Max should be 1.
    y = make_forward_target(usrec, horizon=3)
    assert y.iloc[0] == 1
    # at t=1: lookahead is usrec[2]=0, usrec[3]=1, usrec[4]=0. Max is 1.
    assert y.iloc[1] == 1
    # at t=2: lookahead is usrec[3]=1, usrec[4]=0, usrec[5]=0. Max is 1.
    assert y.iloc[2] == 1
    # at t=3: lookahead is usrec[4]=0, usrec[5]=0, usrec[6]=NaN. Has NaN -> NaN.
    assert np.isnan(y.iloc[3])

def test_no_leakage_sign():
    # Test 4: make sure there is no future leakage via incorrect shift sign (e.g. shift(1) instead of shift(-1))
    usrec = pd.Series([1, 1, 0, 0], index=pd.date_range('2020-01-01', periods=4, freq='MS'))
    # if shifted correctly (-1), at t=2 (2020-03-01), target should look at t=3 (0) and t=4 (NaN) -> NaN.
    # If shifted incorrectly (positive), it might look at past values like t=1 (1) -> 1.
    y = make_forward_target(usrec, horizon=1)
    assert y.iloc[1] == 0  # t=1 looks at t=2 (0)
    assert y.iloc[2] == 0  # t=2 looks at t=3 (0)
    assert np.isnan(y.iloc[3])  # t=3 has no t+1, so it is NaN
