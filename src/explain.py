import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"

# Use representative, low-correlation features for SHAP explainability
# to prevent the multicollinearity distortion of lags/levels.
REPRESENTATIVE_FEATURES = [
    "T10Y3M_level", 
    "T10Y2Y_level", 
    "Credit_Spread_level",
    "Sahm_Indicator", 
    "INDPRO_growth12", 
    "CPI_growth12",
    "FEDFUNDS_level", 
    "SP500_return12", 
    "M2_growth12"
]

FEATURE_LABEL_MAP = {
    "T10Y3M_level": "10Y-3M Term Spread",
    "T10Y2Y_level": "10Y-2Y Term Spread",
    "Credit_Spread_level": "Corporate Credit Spread",
    "Sahm_Indicator": "Sahm Rule Indicator",
    "INDPRO_growth12": "YoY Industrial Production Growth",
    "CPI_growth12": "YoY CPI Inflation",
    "FEDFUNDS_level": "Federal Funds Rate",
    "SP500_return12": "YoY S&P 500 Return",
    "M2_growth12": "YoY M2 Money Supply Growth"
}

# Premium plotting styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'grid.color': '#eeeeee',
    'grid.linewidth': 0.5,
    'figure.titlesize': 14,
    'axes.titlesize': 11,
    'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
})

def generate_shap_plots():
    print("[Explain] Generating SHAP explainability analysis (Figure 3)...")
    processed_path = os.path.join(CWD, "01_data", "processed_monthly_dataset.csv")
    df = pd.read_csv(processed_path, parse_dates=[0], index_col=0)
    df = df.sort_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, horizon in enumerate([3, 6, 12]):
        target_col = f"target_{horizon}m"
        clean_df = df.dropna(subset=[target_col] + REPRESENTATIVE_FEATURES)
        
        X = clean_df[REPRESENTATIVE_FEATURES]
        y = clean_df[target_col]
        
        # Fit a constrained XGBoost model to representative features
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
        xgb_model.fit(X, y)
        
        # Calculate Tree SHAP
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "Feature": [FEATURE_LABEL_MAP[f] for f in REPRESENTATIVE_FEATURES],
            "Mean Absolute SHAP": mean_abs_shap
        }).sort_values(by="Mean Absolute SHAP", ascending=False)
        
        ax = axes[i]
        sns.barplot(
            x="Mean Absolute SHAP",
            y="Feature",
            data=importance_df,
            ax=ax,
            palette="Blues_r",
            edgecolor="#cccccc",
            linewidth=0.8
        )
        
        ax.set_title(f"{horizon}-Month Horizon SHAP Feature Importance", fontweight='bold', fontsize=11)
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Risk Score)", fontsize=9.5)
        ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    fig_dir = os.path.join(CWD, "03_figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    png_path = os.path.join(fig_dir, "figure3_shap.png")
    pdf_path = os.path.join(fig_dir, "figure3_shap.pdf")
    
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Explain] Saved Figure 3 to {png_path} and {pdf_path}")

if __name__ == "__main__":
    generate_shap_plots()
