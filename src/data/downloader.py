import os
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from src.utils.logging_utils import get_logger
from src.data.metadata import update_series_metadata

logger = get_logger("downloader")

FRED_SERIES = [
    "USREC", "PAYEMS", "UNRATE", "FEDFUNDS", "CPIAUCSL", 
    "GS10", "GS5", "TB3MS", "INDPRO", "T10Y2Y", "BAMLH0A0HYM2"
]
YAHOO_SERIES = ["^GSPC"]

def download_fred_series(series_id: str, raw_dir: str) -> str:
    """Download FRED series from public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    logger.info(f"Downloading {series_id} from FRED...")
    
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    
    filepath = os.path.join(raw_dir, f"{series_id}.csv")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    logger.info(f"Saved {series_id} to {filepath}")
    return filepath

def download_yahoo_series(ticker: str, raw_dir: str) -> str:
    """Download Yahoo Finance series using yfinance."""
    logger.info(f"Downloading {ticker} from Yahoo Finance using yfinance...")
    # Get monthly data starting from 1950-01-01
    df = yf.download(ticker, start="1950-01-01", end="2030-01-01", interval="1mo")
    
    # yfinance returns multi-index columns in some newer versions. Let's flatten them if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    # Check column names: yfinance has 'Date' as first column after reset_index
    df.columns = [c.capitalize() if c.lower() == 'date' else c for c in df.columns]
    
    filepath = os.path.join(raw_dir, f"{ticker}.csv")
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {ticker} to {filepath}")
    return filepath

def run_download(config: dict) -> None:
    """Run downloader for all series in configs."""
    base_dir = config.get("outputs", {}).get("base_dir", "outputs/")
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    download_date = datetime.now().strftime("%Y-%m-%d")
    
    for series_id in FRED_SERIES:
        try:
            filepath = download_fred_series(series_id, raw_dir)
            df = pd.read_csv(filepath)
            # Normalize column names to upper case
            df.columns = [c.upper() for c in df.columns]
            
            # Find the date column: could be DATE or OBSERVATION_DATE
            date_col = 'DATE' if 'DATE' in df.columns else ('OBSERVATION_DATE' if 'OBSERVATION_DATE' in df.columns else df.columns[0])
            df = df.rename(columns={date_col: 'DATE'})
            
            # Convert to datetime and sort
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df = df.dropna(subset=['DATE']).sort_values('DATE')
            start_date = df['DATE'].min().strftime("%Y-%m-%d")
            end_date = df['DATE'].max().strftime("%Y-%m-%d")
            
            # Save normalized version back to raw folder
            # Keep original column names but ensure DATE is renamed
            # Let's save it so panel_builder has a consistent header
            # We rename date_col in the raw CSV to DATE
            raw_df = pd.read_csv(filepath)
            # Find index of date_col in raw_df
            for idx, col in enumerate(raw_df.columns):
                if col.upper() in ['DATE', 'OBSERVATION_DATE']:
                    raw_df = raw_df.rename(columns={col: 'DATE'})
                    break
            raw_df.to_csv(filepath, index=False)
            
            update_series_metadata(
                series_id=series_id,
                source="FRED",
                frequency="monthly",
                download_date=download_date,
                start_date=start_date,
                end_date=end_date,
                transformation="none"
            )
        except Exception as e:
            logger.error(f"Error downloading {series_id}: {e}")
            
    for ticker in YAHOO_SERIES:
        try:
            filepath = download_yahoo_series(ticker, raw_dir)
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date')
            start_date = df['Date'].min().strftime("%Y-%m-%d")
            end_date = df['Date'].max().strftime("%Y-%m-%d")
            
            update_series_metadata(
                series_id=ticker,
                source="Yahoo",
                frequency="monthly",
                download_date=download_date,
                start_date=start_date,
                end_date=end_date,
                transformation="none"
            )
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
