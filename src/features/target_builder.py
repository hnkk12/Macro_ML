import pandas as pd
import numpy as np

def make_forward_target(usrec: pd.Series, horizon: int) -> pd.Series:
    """
    y_t = 1 if there is a recession in any month in [t+1, t+horizon].
    - Does not use usrec[t] in the target at t.
    - If there are not enough future months left in the sample (< horizon) -> returns NaN.
    - Output index is forecast_origin (datetime).
    """
    # Shift forward to look ahead from t+1 to t+horizon
    shifted_df = pd.DataFrame({
        f'shift_{i}': usrec.shift(-i) for i in range(1, horizon + 1)
    })
    
    target = shifted_df.max(axis=1)
    
    # If any of the future months are missing (at the end of the series), return NaN
    target[shifted_df.isna().any(axis=1)] = np.nan
    
    target.index = usrec.index
    return target
