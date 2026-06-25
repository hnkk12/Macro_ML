import os
import pandas as pd
import yfinance as yf

CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"

def load_raw_data():
    """
    Loads raw macroeconomic and financial data from the local CSV.
    If the CSV is missing, it attempts to download the series from FRED and Yahoo Finance.
    """
    raw_path = os.path.join(CWD, "01_data", "raw_fred_data.csv")
    if os.path.exists(raw_path):
        print(f"[Data] Loading local raw data from {raw_path}...")
        df = pd.read_csv(raw_path, parse_dates=[0], index_col=0)
        df.index.name = "DATE"
        return df.sort_index()
    
    # Fallback downloader (if the file was deleted or needs replication)
    print("[Data] Local raw data not found. Attempting fallback download...")
    fred_series = {
        "USREC": "USREC",
        "T10Y3M": "T10Y3M",
        "T10Y2Y": "T10Y2Y",
        "UNRATE": "UNRATE",
        "CPIAUCSL": "CPIAUCSL",
        "INDPRO": "INDPRO",
        "FEDFUNDS": "FEDFUNDS",
        "BAA": "BAA",
        "GS10": "GS10",
        "M2SL": "M2SL"
    }
    
    dfs = []
    for name, series_id in fred_series.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            df_temp = pd.read_csv(url, parse_dates=["observation_date"], index_col="observation_date")
            df_temp = df_temp.replace(".", pd.NA)
            df_temp[series_id] = pd.to_numeric(df_temp[series_id], errors="coerce")
            dfs.append(df_temp)
        except Exception as e:
            print(f"[Warning] Failed to download {series_id}: {e}")
            
    if not dfs:
        raise FileNotFoundError("Could not download FRED data and no local cache was found.")
        
    fred_df = dfs[0]
    for df_temp in dfs[1:]:
        fred_df = fred_df.join(df_temp, how="outer")
        
    # S&P 500 Daily Data from Yahoo Finance
    sp500 = yf.download("^GSPC", start="1970-01-01", end="2026-06-01")
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    
    sp500_monthly = sp500["Close"].resample("ME").last().to_frame(name="SP500")
    sp500_monthly.index = sp500_monthly.index.to_period("M").to_timestamp()
    
    fred_monthly = fred_df.resample("MS").mean()
    combined = fred_monthly.join(sp500_monthly, how="outer")
    combined = combined.loc["1980-01-01":"2026-05-01"]
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    combined.to_csv(raw_path)
    return combined
