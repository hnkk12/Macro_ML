import os
import json
import pandas as pd
import yfinance as yf

# Define workspace directory
CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"
os.makedirs(os.path.join(CWD, "01_data"), exist_ok=True)
os.makedirs(os.path.join(CWD, "02_code"), exist_ok=True)

# List of FRED Series IDs to download
fred_series = {
    "USREC": "USREC",          # NBER Recession Indicator
    "T10Y3M": "T10Y3M",        # 10Y - 3M Spread
    "T10Y2Y": "T10Y2Y",        # 10Y - 2Y Spread
    "UNRATE": "UNRATE",        # Unemployment Rate
    "CPIAUCSL": "CPIAUCSL",    # Consumer Price Index
    "INDPRO": "INDPRO",        # Industrial Production Index
    "FEDFUNDS": "FEDFUNDS",    # Federal Funds Rate
    "BAA": "BAA",              # Moody's Seasoned Baa Corporate Bond Yield
    "GS10": "GS10",            # 10-Year Treasury Maturity Rate
    "M2SL": "M2SL"             # M2 Money Supply
}

def download_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"Downloading {series_id} from FRED...")
    df = pd.read_csv(url, parse_dates=["observation_date"], index_col="observation_date")
    # Clean data (some values might be '.' or missing)
    df = df.replace(".", pd.NA)
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df

def download_all_data():
    # Download FRED data
    dfs = []
    for name, series_id in fred_series.items():
        try:
            df = download_fred_series(series_id)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to download {series_id}: {e}")
            
    # Combine FRED series on Date
    fred_df = dfs[0]
    for df in dfs[1:]:
        fred_df = fred_df.join(df, how="outer")
        
    # Download S&P 500 daily data from Yahoo Finance and resample to monthly
    print("Downloading S&P 500 from Yahoo Finance...")
    sp500 = yf.download("^GSPC", start="1970-01-01", end="2026-06-01")
    
    # If sp500 is empty, raise an error or fall back to an existing file
    if sp500.empty:
        print("Warning: yfinance returned empty data. Trying to load from fallback...")
        # Check if we can load S&P 500 from the JSON files
        try:
            fallback = pd.read_json(os.path.join(CWD, "data", "raw", "SP500_pre-cutoff_data.json"))
            fallback["Dates"] = pd.to_datetime(fallback["Dates"])
            sp500_monthly = fallback.set_index("Dates")[["S&P_500_Index"]].rename(columns={"S&P_500_Index": "SP500"})
            print("Successfully loaded fallback S&P 500 data.")
        except Exception as e:
            raise RuntimeError(f"Failed to load S&P 500 fallback: {e}")
    else:
        # Check if S&P 500 has MultiIndex columns (sometimes yfinance returns multi-indexed columns)
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500.columns = sp500.columns.get_level_values(0)
        
        # Aggregate SP500 daily to monthly Close and monthly Average
        # Keep Date index as monthly (end of month or start of month)
        sp500_monthly = sp500["Close"].resample("ME").last().to_frame(name="SP500")
        # Align date to the first of the month to match FRED monthly data
        sp500_monthly.index = sp500_monthly.index.to_period("M").to_timestamp()
    
    # Align FRED data to monthly if it contains daily/weekly observations
    # In our case, FRED monthly data is already on the first of the month
    # Let's filter FRED data and keep only dates starting on the first of the month
    # We will resample all FRED series to first of month using mean
    fred_monthly = fred_df.resample("MS").mean()
    
    # Merge FRED and SP500
    combined = fred_monthly.join(sp500_monthly, how="outer")
    
    # Filter to start from 1980-01-01 to 2026-05-01
    combined = combined.loc["1980-01-01":"2026-05-01"]
    
    # Save raw csv
    raw_path = os.path.join(CWD, "01_data", "raw_fred_data.csv")
    combined.to_csv(raw_path)
    print(f"Saved raw data to {raw_path}")
    print(combined.tail())

if __name__ == "__main__":
    download_all_data()
