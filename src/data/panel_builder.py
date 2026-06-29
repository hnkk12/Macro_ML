import os
import pandas as pd
import numpy as np
from src.utils.logging_utils import get_logger

logger = get_logger("panel_builder")

FRED_SERIES = [
    "USREC", "PAYEMS", "UNRATE", "FEDFUNDS", "CPIAUCSL", 
    "GS10", "GS5", "TB3MS", "INDPRO", "T10Y2Y", "BAMLH0A0HYM2"
]
YAHOO_SERIES = ["^GSPC"]

def build_panel() -> pd.DataFrame:
    """Merge all raw series into a monthly panel."""
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    panel = None
    
    # 1. Process FRED Series
    for series_id in FRED_SERIES:
        filepath = os.path.join(raw_dir, f"{series_id}.csv")
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} not found, skipping.")
            continue
            
        df = pd.read_csv(filepath)
        df.columns = [c.upper() for c in df.columns]
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE'])
        
        # FRED missing values represented as '.'
        df[series_id] = df[series_id].replace(".", np.nan)
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        
        # Monthly alignment: set date to first of the month
        df['DATE'] = df['DATE'].dt.to_period('M').dt.to_timestamp()
        
        # Sort and drop duplicates
        df = df.sort_values('DATE').drop_duplicates(subset=['DATE'])
        df = df[['DATE', series_id]]
        
        if panel is None:
            panel = df
        else:
            panel = pd.merge(panel, df, on='DATE', how='outer')
            
    # 2. Process Yahoo Series
    for ticker in YAHOO_SERIES:
        filepath = os.path.join(raw_dir, f"{ticker}.csv")
        if not os.path.exists(filepath):
            # Try pre-cutoff S&P500 fix as fallback
            filepath = os.path.join(raw_dir, "SP500_pre-cutoff_data.json")
            if os.path.exists(filepath):
                logger.info("Using pre-cutoff S&P 500 JSON data.")
                df_json = pd.read_json(filepath)
                df_json['Dates'] = pd.to_datetime(df_json['Dates'])
                df_json['DATE'] = df_json['Dates'].dt.to_period('M').dt.to_timestamp()
                df_json = df_json.rename(columns={'S&P_500_Index': '^GSPC'})
                df = df_json[['DATE', '^GSPC']]
            else:
                logger.warning(f"S&P 500 file not found, skipping.")
                continue
        else:
            df = pd.read_csv(filepath)
            df.columns = [c.capitalize() if c == 'Date' else c for c in df.columns]
            df['DATE'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['DATE'])
            # Monthly alignment
            df['DATE'] = df['DATE'].dt.to_period('M').dt.to_timestamp()
            # Yahoo CSV has Adj Close column
            adj_col = 'Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else df.columns[1])
            df = df.rename(columns={adj_col: '^GSPC'})
            df = df.sort_values('DATE').drop_duplicates(subset=['DATE'])
            df = df[['DATE', '^GSPC']]
            
        if panel is None:
            panel = df
        else:
            panel = pd.merge(panel, df, on='DATE', how='outer')
            
    if panel is not None:
        panel = panel.sort_values('DATE').reset_index(drop=True)
        # Drop rows where DATE is null
        panel = panel.dropna(subset=['DATE'])
        
        # Interpolate any missing values (linear interpolation)
        panel = panel.interpolate(method='linear', limit_direction='both')
        
        out_path = os.path.join(processed_dir, "panel_revised.csv")
        panel.to_csv(out_path, index=False)
        logger.info(f"Saved merged panel to {out_path} (shape: {panel.shape})")
        return panel
    else:
        logger.error("No data series merged.")
        return pd.DataFrame()
