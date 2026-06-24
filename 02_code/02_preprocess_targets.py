import os
import numpy as np
import pandas as pd

# Define workspace directory
CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"

def build_features_and_targets():
    # Load raw data
    raw_path = os.path.join(CWD, "01_data", "raw_fred_data.csv")
    df = pd.read_csv(raw_path, parse_dates=[0], index_col=0)
    df.index.name = "DATE"
    
    # Sort chronological order
    df = df.sort_index()
    
    # Build targets
    # y_t^h = 1 if USREC is 1 in any month t+1 to t+h
    # y_t^h = 0 if USREC is 0 in all months t+1 to t+h
    # We use reverse rolling max to look forward in time, then shift back by 1 to start from t+1
    for h in [3, 6, 12]:
        df[f"target_{h}m"] = df["USREC"].iloc[::-1].rolling(window=h, min_periods=h).max().iloc[::-1].shift(-1)
    
    # Feature Engineering
    features = pd.DataFrame(index=df.index)
    
    # Keep raw USREC for target reference, but do NOT use as feature
    # 1. Yield curve spreads
    features["T10Y3M_level"] = df["T10Y3M"]
    features["T10Y2Y_level"] = df["T10Y2Y"]
    
    # Lags for spreads
    for lag in [1, 3, 6, 12]:
        features[f"T10Y3M_lag{lag}"] = df["T10Y3M"].shift(lag)
        features[f"T10Y2Y_lag{lag}"] = df["T10Y2Y"].shift(lag)
        
    # 2. Labor market
    features["UNRATE_level"] = df["UNRATE"]
    features["UNRATE_diff3"] = df["UNRATE"] - df["UNRATE"].shift(3)
    features["UNRATE_diff12"] = df["UNRATE"] - df["UNRATE"].shift(12)
    
    # Sahm Rule Indicator: 3-month moving average of unemployment minus the minimum 3-month MA in the past 12 months
    unrate_ma3 = df["UNRATE"].rolling(3, min_periods=3).mean()
    unrate_min12 = unrate_ma3.shift(1).rolling(12, min_periods=12).min()
    features["Sahm_Indicator"] = unrate_ma3 - unrate_min12
    
    # 3. Inflation/Price
    # YoY growth rate (log difference)
    features["CPI_YoY"] = np.log(df["CPIAUCSL"] / df["CPIAUCSL"].shift(12))
    features["CPI_MoM"] = np.log(df["CPIAUCSL"] / df["CPIAUCSL"].shift(1))
    
    # 4. Real activity
    features["INDPRO_YoY"] = np.log(df["INDPRO"] / df["INDPRO"].shift(12))
    features["INDPRO_MoM"] = np.log(df["INDPRO"] / df["INDPRO"].shift(1))
    
    # 5. Monetary policy
    features["FEDFUNDS_level"] = df["FEDFUNDS"]
    features["FEDFUNDS_diff1"] = df["FEDFUNDS"] - df["FEDFUNDS"].shift(1)
    features["FEDFUNDS_diff12"] = df["FEDFUNDS"] - df["FEDFUNDS"].shift(12)
    
    # 6. Credit market
    # Moody's Baa minus GS10 spread
    df["Credit_Spread"] = df["BAA"] - df["GS10"]
    features["Credit_Spread_level"] = df["Credit_Spread"]
    for lag in [1, 3, 6, 12]:
        features[f"Credit_Spread_lag{lag}"] = df["Credit_Spread"].shift(lag)
        
    # 7. Stock market
    # S&P 500 log returns
    features["SP500_Return_1m"] = np.log(df["SP500"] / df["SP500"].shift(1))
    features["SP500_Return_3m"] = np.log(df["SP500"] / df["SP500"].shift(3))
    features["SP500_Return_12m"] = np.log(df["SP500"] / df["SP500"].shift(12))
    
    # S&P 500 Drawdown proxy
    running_max = df["SP500"].cummax()
    features["SP500_Drawdown"] = (df["SP500"] - running_max) / running_max
    
    # 8. Money supply
    features["M2_YoY"] = np.log(df["M2SL"] / df["M2SL"].shift(12))
    features["M2_MoM"] = np.log(df["M2SL"] / df["M2SL"].shift(1))
    
    # Add target columns to features DataFrame
    processed_df = features.copy()
    processed_df["target_3m"] = df["target_3m"]
    processed_df["target_6m"] = df["target_6m"]
    processed_df["target_12m"] = df["target_12m"]
    processed_df["USREC"] = df["USREC"]  # Keep raw USREC label for reference
    
    # Drop rows where target is NaN (at the end of the sample because of looking forward)
    # We also drop rows where features are NaN (at the start of the sample because of lags)
    # This prevents any leakage or missing values.
    # Note: We must be careful not to drop the test set when evaluating.
    # For target_12m, the last 12 months will have target as NaN.
    # This is natural since we cannot evaluate target_12m for recent data.
    # Let's save the full dataframe, and when training/testing we will filter out rows with NaN in features/targets.
    
    # Save processed monthly dataset
    processed_path = os.path.join(CWD, "01_data", "processed_monthly_dataset.csv")
    processed_df.to_csv(processed_path)
    print(f"Saved processed dataset to {processed_path}")
    print("Features shape:", features.shape)
    print(processed_df.tail(15))
    
    # Create Data Dictionary
    dictionary_data = [
        {"Variable": "target_3m", "Source": "FRED (USREC)", "Transformation": "Max of USREC from t+1 to t+3", "Frequency": "Monthly", "Description": "Recession indicator within next 3 months"},
        {"Variable": "target_6m", "Source": "FRED (USREC)", "Transformation": "Max of USREC from t+1 to t+6", "Frequency": "Monthly", "Description": "Recession indicator within next 6 months"},
        {"Variable": "target_12m", "Source": "FRED (USREC)", "Transformation": "Max of USREC from t+1 to t+12", "Frequency": "Monthly", "Description": "Recession indicator within next 12 months"},
        {"Variable": "T10Y3M_level", "Source": "FRED (T10Y3M)", "Transformation": "Level", "Frequency": "Monthly", "Description": "10-Year minus 3-Month Treasury spread"},
        {"Variable": "T10Y2Y_level", "Source": "FRED (T10Y2Y)", "Transformation": "Level", "Frequency": "Monthly", "Description": "10-Year minus 2-Year Treasury spread"},
        {"Variable": "UNRATE_level", "Source": "FRED (UNRATE)", "Transformation": "Level", "Frequency": "Monthly", "Description": "Civilian Unemployment Rate"},
        {"Variable": "UNRATE_diff3", "Source": "FRED (UNRATE)", "Transformation": "3-month difference", "Frequency": "Monthly", "Description": "Unemployment rate 3-month change"},
        {"Variable": "UNRATE_diff12", "Source": "FRED (UNRATE)", "Transformation": "12-month difference", "Frequency": "Monthly", "Description": "Unemployment rate 12-month change"},
        {"Variable": "Sahm_Indicator", "Source": "FRED (UNRATE)", "Transformation": "3m moving average minus 12m minimum", "Frequency": "Monthly", "Description": "Sahm Rule recession indicator"},
        {"Variable": "CPI_YoY", "Source": "FRED (CPIAUCSL)", "Transformation": "12-month log difference", "Frequency": "Monthly", "Description": "Year-over-Year Consumer Price Index inflation"},
        {"Variable": "CPI_MoM", "Source": "FRED (CPIAUCSL)", "Transformation": "1-month log difference", "Frequency": "Monthly", "Description": "Month-over-Month Consumer Price Index inflation"},
        {"Variable": "INDPRO_YoY", "Source": "FRED (INDPRO)", "Transformation": "12-month log difference", "Frequency": "Monthly", "Description": "Year-over-Year Industrial Production Index growth"},
        {"Variable": "INDPRO_MoM", "Source": "FRED (INDPRO)", "Transformation": "1-month log difference", "Frequency": "Monthly", "Description": "Month-over-Month Industrial Production Index growth"},
        {"Variable": "FEDFUNDS_level", "Source": "FRED (FEDFUNDS)", "Transformation": "Level", "Frequency": "Monthly", "Description": "Effective Federal Funds Rate"},
        {"Variable": "FEDFUNDS_diff1", "Source": "FRED (FEDFUNDS)", "Transformation": "1-month difference", "Frequency": "Monthly", "Description": "Federal Funds Rate 1-month change"},
        {"Variable": "FEDFUNDS_diff12", "Source": "FRED (FEDFUNDS)", "Transformation": "12-month difference", "Frequency": "Monthly", "Description": "Federal Funds Rate 12-month change"},
        {"Variable": "Credit_Spread_level", "Source": "FRED (BAA, GS10)", "Transformation": "BAA minus GS10", "Frequency": "Monthly", "Description": "Moody's Baa corporate bond yield relative to 10Y Treasury"},
        {"Variable": "SP500_Return_1m", "Source": "Yahoo Finance (^GSPC)", "Transformation": "1-month log difference", "Frequency": "Monthly", "Description": "S&P 500 1-month return"},
        {"Variable": "SP500_Return_3m", "Source": "Yahoo Finance (^GSPC)", "Transformation": "3-month log difference", "Frequency": "Monthly", "Description": "S&P 500 3-month return"},
        {"Variable": "SP500_Return_12m", "Source": "Yahoo Finance (^GSPC)", "Transformation": "12-month log difference", "Frequency": "Monthly", "Description": "S&P 500 Year-over-Year return"},
        {"Variable": "SP500_Drawdown", "Source": "Yahoo Finance (^GSPC)", "Transformation": "Drawdown from running peak", "Frequency": "Monthly", "Description": "S&P 500 Drawdown percentage"},
        {"Variable": "M2_YoY", "Source": "FRED (M2SL)", "Transformation": "12-month log difference", "Frequency": "Monthly", "Description": "Year-over-Year M2 Money Supply growth"},
        {"Variable": "M2_MoM", "Source": "FRED (M2SL)", "Transformation": "1-month log difference", "Frequency": "Monthly", "Description": "Month-over-Month M2 Money Supply growth"}
    ]
    
    # Add lags of spreads and credit spread to dictionary
    for lag in [1, 3, 6, 12]:
        dictionary_data.append({"Variable": f"T10Y3M_lag{lag}", "Source": "FRED (T10Y3M)", "Transformation": f"{lag}-month lag", "Frequency": "Monthly", "Description": f"10Y-3M Treasury spread {lag}-month lag"})
        dictionary_data.append({"Variable": f"T10Y2Y_lag{lag}", "Source": "FRED (T10Y2Y)", "Transformation": f"{lag}-month lag", "Frequency": "Monthly", "Description": f"10Y-2Y Treasury spread {lag}-month lag"})
        dictionary_data.append({"Variable": f"Credit_Spread_lag{lag}", "Source": "FRED (BAA, GS10)", "Transformation": f"{lag}-month lag", "Frequency": "Monthly", "Description": f"Credit spread {lag}-month lag"})
        
    dict_df = pd.DataFrame(dictionary_data)
    dict_path = os.path.join(CWD, "01_data", "data_dictionary.csv")
    dict_df.to_csv(dict_path, index=False)
    print(f"Saved data dictionary to {dict_path}")
    
    # Also save to xlsx format for the tables folder as Table 1
    tables_dir = os.path.join(CWD, "04_tables")
    os.makedirs(tables_dir, exist_ok=True)
    table1_path = os.path.join(tables_dir, "table1_variables.xlsx")
    dict_df.to_excel(table1_path, index=False)
    print(f"Saved variable table to {table1_path}")

if __name__ == "__main__":
    build_features_and_targets()
