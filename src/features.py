import os
import numpy as np
import pandas as pd

CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"

def engineer_features(df):
    """
    Engineers a high-dimensional macro-financial feature panel from the raw indicators.
    Total features: ~80 features covering spreads, momentum, volatilities, and growth rates.
    """
    features = pd.DataFrame(index=df.index)
    
    # ------------------ 1. Interest Rate & Yield Spreads ------------------
    features["T10Y3M_level"] = df["T10Y3M"]
    features["T10Y2Y_level"] = df["T10Y2Y"]
    
    # Credit spread (Moody's Baa corporate bond yield minus 10-year Treasury rate)
    credit_spread = df["BA"] - df["GS10"] if "BA" in df.columns else df["BAA"] - df["GS10"]
    features["Credit_Spread_level"] = credit_spread
    
    # Lags for spreads (1, 2, 3, 6, 12 months)
    for lag in [1, 2, 3, 6, 12]:
        features[f"T10Y3M_lag{lag}"] = df["T10Y3M"].shift(lag)
        features[f"T10Y2Y_lag{lag}"] = df["T10Y2Y"].shift(lag)
        features[f"Credit_Spread_lag{lag}"] = credit_spread.shift(lag)
        
    # Rate momentum (differences over 1, 3, 6, 12 months)
    for diff in [1, 3, 6, 12]:
        features[f"T10Y3M_diff{diff}"] = df["T10Y3M"] - df["T10Y3M"].shift(diff)
        features[f"T10Y2Y_diff{diff}"] = df["T10Y2Y"] - df["T10Y2Y"].shift(diff)
        features[f"Credit_Spread_diff{diff}"] = credit_spread - credit_spread.shift(diff)
        
    # Volatilities (rolling standard deviations)
    for window in [3, 6, 12]:
        features[f"T10Y3M_vol{window}"] = df["T10Y3M"].rolling(window).std()
        features[f"T10Y2Y_vol{window}"] = df["T10Y2Y"].rolling(window).std()
        features[f"Credit_Spread_vol{window}"] = credit_spread.rolling(window).std()

    # ------------------ 2. Labor Market ------------------
    features["UNRATE_level"] = df["UNRATE"]
    for diff in [1, 3, 6, 12]:
        features[f"UNRATE_diff{diff}"] = df["UNRATE"] - df["UNRATE"].shift(diff)
        
    # Sahm Rule Indicator (3m MA minus 12m minimum)
    unrate_ma3 = df["UNRATE"].rolling(3, min_periods=3).mean()
    unrate_min12 = unrate_ma3.shift(1).rolling(12, min_periods=12).min()
    features["Sahm_Indicator"] = unrate_ma3 - unrate_min12
    
    # ------------------ 3. Production and Real Output ------------------
    features["INDPRO_level"] = df["INDPRO"]
    for growth in [1, 3, 6, 12]:
        features[f"INDPRO_growth{growth}"] = np.log(df["INDPRO"] / df["INDPRO"].shift(growth))
        
    for window in [3, 6, 12]:
        features[f"INDPRO_vol{window}"] = features["INDPRO_growth1"].rolling(window).std()
        
    # ------------------ 4. Inflation ------------------
    features["CPI_level"] = df["CPIAUCSL"]
    for growth in [1, 3, 6, 12]:
        features[f"CPI_growth{growth}"] = np.log(df["CPIAUCSL"] / df["CPIAUCSL"].shift(growth))
        
    features["CPI_acceleration"] = features["CPI_growth1"] - features["CPI_growth1"].shift(1)
    
    # ------------------ 5. Monetary Policy ------------------
    features["FEDFUNDS_level"] = df["FEDFUNDS"]
    for diff in [1, 3, 6, 12]:
        features[f"FEDFUNDS_diff{diff}"] = df["FEDFUNDS"] - df["FEDFUNDS"].shift(diff)
        
    # ------------------ 6. Stock Market ------------------
    features["SP500_level"] = df["SP500"]
    for return_p in [1, 3, 6, 12]:
        features[f"SP500_return{return_p}"] = np.log(df["SP500"] / df["SP500"].shift(return_p))
        
    for window in [3, 6, 12]:
        features[f"SP500_vol{window}"] = features["SP500_return1"].rolling(window).std()
        
    running_max = df["SP500"].cummax()
    features["SP500_Drawdown"] = (df["SP500"] - running_max) / running_max
    
    # ------------------ 7. Money Supply ------------------
    features["M2_level"] = df["M2SL"]
    for growth in [1, 3, 6, 12]:
        features[f"M2_growth{growth}"] = np.log(df["M2SL"] / df["M2SL"].shift(growth))
        
    return features

def prepare_targets(df):
    """
    Constructs the future-looking multi-horizon targets (3M, 6M, 12M).
    y_t^h = 1 if USREC = 1 in any month from t+1 to t+h, else 0.
    """
    targets = pd.DataFrame(index=df.index)
    for h in [3, 6, 12]:
        # Reverse rolling max to look forward, then shift back by 1
        targets[f"target_{h}m"] = df["USREC"].iloc[::-1].rolling(window=h, min_periods=h).max().iloc[::-1].shift(-1)
    return targets

def run_feature_pipeline():
    """
    Main entry point for feature engineering and target construction.
    Loads raw data, computes features and targets, and saves to 01_data.
    """
    from src.data_processing import load_raw_data
    df = load_raw_data()
    
    # Generate targets and features
    targets = prepare_targets(df)
    features = engineer_features(df)
    
    processed_df = features.copy()
    for col in targets.columns:
        processed_df[col] = targets[col]
    processed_df["USREC"] = df["USREC"]
    
    # Save the processed dataset
    processed_path = os.path.join(CWD, "01_data", "processed_monthly_dataset.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    processed_df.to_csv(processed_path)
    print(f"[Features] Saved engineered features panel ({processed_df.shape}) to {processed_path}")
    
    # Generate Data Dictionary
    dictionary_data = [
        {"Variable": "target_3m", "Source": "NBER / FRED", "Transformation": "Forward Rolling Max (t+1 to t+3)", "Description": "Recession occurring within the next 3 months"},
        {"Variable": "target_6m", "Source": "NBER / FRED", "Transformation": "Forward Rolling Max (t+1 to t+6)", "Description": "Recession occurring within the next 6 months"},
        {"Variable": "target_12m", "Source": "NBER / FRED", "Transformation": "Forward Rolling Max (t+1 to t+12)", "Description": "Recession occurring within the next 12 months"},
        {"Variable": "T10Y3M_level", "Source": "FRED (T10Y3M)", "Transformation": "Level", "Description": "10-Year Treasury Constant Maturity minus 3-Month Treasury Constant Maturity Yield Spread"},
        {"Variable": "T10Y2Y_level", "Source": "FRED (T10Y2Y)", "Transformation": "Level", "Description": "10-Year Treasury Constant Maturity minus 2-Year Treasury Constant Maturity Yield Spread"},
        {"Variable": "Credit_Spread_level", "Source": "FRED (BAA/GS10)", "Transformation": "Spread (BAA - GS10)", "Description": "Moody's Seasoned Baa Corporate Bond Yield relative to 10Y Treasury maturity rate"},
        {"Variable": "UNRATE_level", "Source": "FRED (UNRATE)", "Transformation": "Level", "Description": "Civilian Unemployment Rate"},
        {"Variable": "Sahm_Indicator", "Source": "FRED (UNRATE)", "Transformation": "Moving average difference", "Description": "Sahm Rule recession indicator (3m MA minus 12m minimum)"},
        {"Variable": "INDPRO_growth12", "Source": "FRED (INDPRO)", "Transformation": "12-month log difference", "Description": "Year-over-Year Industrial Production growth rate"},
        {"Variable": "CPI_growth12", "Source": "FRED (CPIAUCSL)", "Transformation": "12-month log difference", "Description": "Year-over-Year Consumer Price Index inflation rate"},
        {"Variable": "FEDFUNDS_level", "Source": "FRED (FEDFUNDS)", "Transformation": "Level", "Description": "Effective Federal Funds Rate"},
        {"Variable": "SP500_return12", "Source": "Yahoo Finance (^GSPC)", "Transformation": "12-month log difference", "Description": "S&P 500 Year-over-Year log return"},
        {"Variable": "SP500_Drawdown", "Source": "Yahoo Finance (^GSPC)", "Transformation": "Peak-to-trough drawdown", "Description": "S&P 500 Drawdown percentage from cumulative peak"},
        {"Variable": "M2_growth12", "Source": "FRED (M2SL)", "Transformation": "12-month log difference", "Description": "Year-over-Year M2 Money Supply growth rate"}
    ]
    
    # Save Data Dictionary to tables folder (Table 1) and data folder
    dict_df = pd.DataFrame(dictionary_data)
    dict_path = os.path.join(CWD, "01_data", "data_dictionary.csv")
    dict_df.to_csv(dict_path, index=False)
    
    tables_dir = os.path.join(CWD, "04_tables")
    os.makedirs(tables_dir, exist_ok=True)
    dict_df.to_excel(os.path.join(tables_dir, "table1_variables.xlsx"), index=False)
    print(f"[Features] Saved Variable Dictionary (Table 1) to {dict_path} and 04_tables/table1_variables.xlsx")

if __name__ == "__main__":
    run_feature_pipeline()
