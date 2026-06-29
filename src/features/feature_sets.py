import os
import pandas as pd
import numpy as np
from typing import List, Dict
from src.utils.logging_utils import get_logger

logger = get_logger("feature_sets")

FEATURE_SETS = {
    "full": [
        "Payrolls_3mo_vs_12mo", "Real_Fed_Funds_Rate_12mo_chg",
        "CPI_3mo_pct_chg_annualized", "10Y_Treasury_Rate_12mo_chg",
        "3M_10Y_Treasury_Spread", "S&P_500_12mo_chg",
        "Payrolls_3mo_pct_chg_annualized", "Payrolls_12mo_pct_chg",
        "Unemployment_Rate", "Unemployment_Rate_12mo_chg",
        "Real_Fed_Funds_Rate", "CPI_12mo_pct_chg",
        "CPI_3mo_vs_12mo", "3M_Treasury_Rate_12mo_chg",
        "3M_10Y_Treasury_Spread_12mo_chg", "5Y_10Y_Treasury_Spread",
        "S&P_500_3mo_chg", "S&P_500_3mo_vs_12mo",
        "IPI_3mo_pct_chg_annualized", "IPI_12mo_pct_chg",
        "IPI_3mo_vs_12mo", "T10Y2Y", "BAMLH0A0HYM2"
    ],
    "core_econometric": [
        "T10Y3M",
        "T10Y2Y",
        "UNRATE_chg_12mo",
        "BAMLH0A0HYM2"
    ],
    "yield_only": [
        "T10Y3M",
        "T10Y2Y"
    ],
    "low_revision": [
        "T10Y3M",
        "T10Y2Y",
        "FEDFUNDS",
        "BAMLH0A0HYM2",
        "SP500_12mo_return",
        "SP500_drawdown_6mo"
    ]
}

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 21+ features from panel data.
    Input df columns: DATE, USREC, PAYEMS, UNRATE, FEDFUNDS, CPIAUCSL, GS10, GS5, TB3MS, INDPRO, T10Y2Y, BAMLH0A0HYM2, ^GSPC
    """
    df = df.sort_values('DATE').reset_index(drop=True)
    out = pd.DataFrame()
    out['DATE'] = df['DATE']
    
    payems = df['PAYEMS']
    unrate = df['UNRATE']
    fedfunds = df['FEDFUNDS']
    cpi = df['CPIAUCSL']
    gs10 = df['GS10']
    gs5 = df['GS5']
    tb3ms = df['TB3MS']
    indpro = df['INDPRO']
    t10y2y_fred = df['T10Y2Y']
    baml_spread = df['BAMLH0A0HYM2']
    sp500 = df['^GSPC']
    
    # 1. Payrolls
    payrolls_3mo_pct = (payems / payems.shift(3)) - 1
    out['Payrolls_3mo_pct_chg_annualized'] = ((1 + payrolls_3mo_pct) ** 4) - 1
    out['Payrolls_12mo_pct_chg'] = (payems / payems.shift(12)) - 1
    out['Payrolls_3mo_vs_12mo'] = out['Payrolls_3mo_pct_chg_annualized'] - out['Payrolls_12mo_pct_chg']
    
    # 2. Unemployment
    out['Unemployment_Rate'] = unrate
    out['Unemployment_Rate_12mo_chg'] = unrate - unrate.shift(12)
    out['UNRATE_chg_12mo'] = out['Unemployment_Rate_12mo_chg']
    
    # 3. CPI Inflation
    cpi_3mo_pct = (cpi / cpi.shift(3)) - 1
    out['CPI_3mo_pct_chg_annualized'] = ((1 + cpi_3mo_pct) ** 4) - 1
    out['CPI_12mo_pct_chg'] = (cpi / cpi.shift(12)) - 1
    out['CPI_3mo_vs_12mo'] = out['CPI_3mo_pct_chg_annualized'] - out['CPI_12mo_pct_chg']
    
    # 4. Interest Rates / Yield Spreads
    out['Real_Fed_Funds_Rate'] = fedfunds - (out['CPI_12mo_pct_chg'] * 100)
    out['Real_Fed_Funds_Rate_12mo_chg'] = out['Real_Fed_Funds_Rate'] - out['Real_Fed_Funds_Rate'].shift(12)
    out['FEDFUNDS'] = fedfunds
    
    out['10Y_Treasury_Rate_12mo_chg'] = gs10 - gs10.shift(12)
    out['3M_Treasury_Rate_12mo_chg'] = tb3ms - tb3ms.shift(12)
    
    # Yield spreads
    out['T10Y3M'] = gs10 - tb3ms
    out['3M_10Y_Treasury_Spread'] = out['T10Y3M']
    out['3M_10Y_Treasury_Spread_12mo_chg'] = out['3M_10Y_Treasury_Spread'] - out['3M_10Y_Treasury_Spread'].shift(12)
    out['T10Y2Y'] = t10y2y_fred
    out['5Y_10Y_Treasury_Spread'] = gs10 - gs5
    
    # Credit spread
    out['BAMLH0A0HYM2'] = baml_spread
    
    # 5. S&P 500
    out['S&P_500_3mo_chg'] = (sp500 / sp500.shift(3)) - 1
    out['S&P_500_12mo_chg'] = (sp500 / sp500.shift(12)) - 1
    out['S&P_500_3mo_vs_12mo'] = out['S&P_500_3mo_chg'] - out['S&P_500_12mo_chg']
    out['SP500_12mo_return'] = out['S&P_500_12mo_chg']
    
    rolling_max_6mo = sp500.rolling(window=6, min_periods=1).max()
    out['SP500_drawdown_6mo'] = (sp500 - rolling_max_6mo) / rolling_max_6mo
    
    # 6. Industrial Production (IPI)
    ipi_3mo_pct = (indpro / indpro.shift(3)) - 1
    out['IPI_3mo_pct_chg_annualized'] = ((1 + ipi_3mo_pct) ** 4) - 1
    out['IPI_12mo_pct_chg'] = (indpro / indpro.shift(12)) - 1
    out['IPI_3mo_vs_12mo'] = out['IPI_3mo_pct_chg_annualized'] - out['IPI_12mo_pct_chg']
    
    # Final cleanup: forward fill any NaN values to make sure there are no holes in features, then backfill
    out = out.ffill().bfill()
    
    return out

def get_feature_set_columns(name: str) -> List[str]:
    """Get list of columns for a named feature set."""
    return FEATURE_SETS.get(name, [])

def save_feature_dictionary(output_path: str) -> None:
    """Generate and save the feature dictionary CSV."""
    rows = []
    # Loop over all unique features across all sets
    all_features = set()
    for s in FEATURE_SETS.values():
        all_features.update(s)
        
    for f in sorted(list(all_features)):
        desc = f.replace("_", " ")
        source = "FRED"
        if "SP500" in f or "S&P" in f:
            source = "Yahoo Finance"
        elif "Payrolls" in f:
            source = "FRED (PAYEMS)"
        elif "Unemployment" in f or "UNRATE" in f:
            source = "FRED (UNRATE)"
        elif "CPI" in f:
            source = "FRED (CPIAUCSL)"
        elif "Treasury" in f or "T10Y" in f or "FEDFUNDS" in f or "GS" in f or "TB3" in f:
            source = "FRED (Interest Rates)"
        elif "BAML" in f:
            source = "FRED (BAML Credit Spreads)"
        elif "IPI" in f:
            source = "FRED (INDPRO)"
            
        rows.append({
            "Feature Name": f,
            "Description": desc,
            "Source": source
        })
        
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved feature dictionary to {output_path}")
