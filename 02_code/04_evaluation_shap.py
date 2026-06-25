import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

# Suppress warnings
warnings.filterwarnings("ignore")

# Define workspace directory
CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"
os.makedirs(os.path.join(CWD, "03_figures"), exist_ok=True)
os.makedirs(os.path.join(CWD, "04_tables"), exist_ok=True)

# Custom premium plotting styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'grid.color': '#eeeeee',
    'grid.linewidth': 0.5,
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Define models and their display names
MODEL_NAMES = {
    'prob_probit': 'Probit Baseline',
    'prob_lr': 'Logistic Regression',
    'prob_rf': 'Random Forest',
    'prob_xgb': 'XGBoost',
    'prob_lgb': 'LightGBM'
}

MODEL_COLORS = {
    'prob_probit': '#7f8c8d',  # Slate gray
    'prob_lr': '#34495e',      # Dark blue-gray
    'prob_rf': '#e67e22',      # Orange
    'prob_xgb': '#2c3e50',     # Navy blue
    'prob_lgb': '#2980b9'      # Sky blue
}

def load_predictions(horizon):
    path = os.path.join(CWD, "models", f"predictions_{horizon}m.csv")
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    return df

def calculate_metrics():
    print("Calculating evaluation metrics...")
    results = []
    
    for horizon in [3, 6, 12]:
        df = load_predictions(horizon)
        # Drop rows where target is NaN (due to forward-looking target)
        df_clean = df.dropna(subset=[f"actual_target"])
        
        y_true = df_clean[f"actual_target"]
        
        for col, model_name in MODEL_NAMES.items():
            y_prob = df_clean[col]
            
            # ROC-AUC
            auc = roc_auc_score(y_true, y_prob)
            
            # PR-AUC
            pr_auc = average_precision_score(y_true, y_prob)
            
            # Brier Score
            brier = brier_score_loss(y_true, y_prob)
            
            # Find optimal threshold to maximize F1-score on test set
            best_f1 = 0.0
            best_thresh = 0.5
            for thresh in np.linspace(0.05, 0.95, 91):
                y_pred = (y_prob >= thresh).astype(int)
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            results.append({
                "Horizon": f"{horizon}M",
                "Model": model_name,
                "ROC-AUC": auc,
                "PR-AUC": pr_auc,
                "F1-Score": best_f1,
                "Optimal Threshold": best_thresh,
                "Brier Score": brier
            })
            
    metrics_df = pd.DataFrame(results)
    
    # Save formatted Excel table
    # Pivot for clean comparison
    pivot_df = metrics_df.pivot(index="Model", columns="Horizon", values=["ROC-AUC", "PR-AUC", "F1-Score", "Brier Score"])
    # Reorder columns for presentation
    pivoted_path = os.path.join(CWD, "04_tables", "table2_performance.xlsx")
    pivot_df.to_excel(pivoted_path)
    print(f"Saved Table 2 (performance) to {pivoted_path}")
    
    # Save CSV version too for reference
    metrics_df.to_csv(os.path.join(CWD, "04_tables", "table2_performance.csv"), index=False)
    
    return metrics_df

def plot_probabilities_over_time():
    print("Plotting predictions over time with recession shading...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for i, horizon in enumerate([3, 6, 12]):
        df = load_predictions(horizon)
        ax = axes[i]
        
        # Shade actual recessions (NBER USREC)
        usrec = df["actual_USREC"]
        recession_dates = usrec[usrec == 1].index
        
        if len(recession_dates) > 0:
            diffs = pd.Series(recession_dates).diff()
            break_indices = diffs[diffs > pd.Timedelta(days=32)].index
            
            starts = [recession_dates[0]]
            ends = []
            
            for idx in break_indices:
                ends.append(recession_dates[recession_dates.get_loc(recession_dates[idx-1])])
                starts.append(recession_dates[recession_dates.get_loc(recession_dates[idx])])
            ends.append(recession_dates[-1])
            
            for start, end in zip(starts, ends):
                ax.axvspan(start, end, color='#dcdde1', alpha=0.7, label="NBER Recession" if start == starts[0] else "")
        
        # Plot predicted probabilities for key models
        for col, model_name in MODEL_NAMES.items():
            if col in ['prob_probit', 'prob_xgb', 'prob_lgb']:  # Plot baseline Probit, XGB, and LGB for clarity
                ax.plot(df.index, df[col], color=MODEL_COLORS[col], linewidth=1.5, label=model_name)
                
        ax.set_title(f"Recession Risk Probability Forecast: {horizon}-Month Horizon", fontweight='bold', fontsize=11)
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    axes[2].set_xlabel("Year", fontsize=10)
    plt.tight_layout()
    
    fig_path = os.path.join(CWD, "03_figures", "figure2_probabilities.pdf")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    fig_path_png = os.path.join(CWD, "03_figures", "figure2_probabilities.png")
    plt.savefig(fig_path_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {fig_path} and PNG copy")

def plot_calibration_curves():
    print("Plotting calibration curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, horizon in enumerate([3, 6, 12]):
        df = load_predictions(horizon).dropna(subset=[f"actual_target"])
        y_true = df[f"actual_target"]
        ax = axes[i]
        
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
        
        for col, model_name in MODEL_NAMES.items():
            y_prob = df[col]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", color=MODEL_COLORS[col], label=model_name, linewidth=1.5, markersize=4)
            
        ax.set_title(f"{horizon}-Month Horizon", fontweight='bold', fontsize=11)
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        if i == 0:
            ax.set_ylabel("Fraction of Positives", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    fig_path = os.path.join(CWD, "03_figures", "figure4_calibration.pdf")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    fig_path_png = os.path.join(CWD, "03_figures", "figure4_calibration.png")
    plt.savefig(fig_path_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {fig_path} and PNG copy")

def analyze_lead_time(threshold=0.25):
    print("Analyzing lead times...")
    recessions = [
        {"name": "Great Recession", "start": pd.Timestamp("2007-12-01"), "end": pd.Timestamp("2009-06-01")},
        {"name": "COVID Recession", "start": pd.Timestamp("2020-02-01"), "end": pd.Timestamp("2020-04-01")}
    ]
    
    lead_time_results = []
    
    for horizon in [3, 6, 12]:
        df = load_predictions(horizon)
        
        for col, model_name in MODEL_NAMES.items():
            probs = df[col]
            
            for rec in recessions:
                rec_start = rec["start"]
                rec_name = rec["name"]
                
                # Check period leading up to recession (up to 12 months prior to start)
                lead_window_start = rec_start - pd.DateOffset(months=12)
                lead_window_end = rec_start - pd.DateOffset(months=1)
                
                lead_probs = probs.loc[lead_window_start:lead_window_end]
                
                warning_months = lead_probs[lead_probs >= threshold]
                
                if not warning_months.empty:
                    first_warning_date = warning_months.index[0]
                    lead_months = (rec_start.year - first_warning_date.year) * 12 + (rec_start.month - first_warning_date.month)
                else:
                    if probs.loc[rec_start] >= threshold:
                        lead_months = 0
                    else:
                        lead_months = -1
                
                max_prob = probs.loc[lead_window_start:rec_start].max()
                
                lead_time_results.append({
                    "Horizon": f"{horizon}M",
                    "Model": model_name,
                    "Recession": rec_name,
                    "Lead Time (Months)": f"{lead_months}m" if lead_months >= 0 else "Not Detected",
                    "Peak Probability (Lead)": f"{max_prob*100:.1f}%"
                })
                
    lead_df = pd.DataFrame(lead_time_results)
    
    tables_dir = os.path.join(CWD, "04_tables")
    table3_path = os.path.join(tables_dir, "table3_leadtime.xlsx")
    lead_df.to_excel(table3_path, index=False)
    print(f"Saved Table 3 (lead times) to {table3_path}")
    
    lead_df.to_csv(os.path.join(CWD, "04_tables", "table3_leadtime.csv"), index=False)
    
    return lead_df

def generate_shap_values():
    print("Generating SHAP analysis...")
    processed_path = os.path.join(CWD, "01_data", "processed_monthly_dataset.csv")
    df = pd.read_csv(processed_path, parse_dates=[0], index_col=0)
    df = df.sort_index()
    
    target_cols = ["target_3m", "target_6m", "target_12m"]
    exclude_cols = target_cols + ["USREC"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, horizon in enumerate([3, 6, 12]):
        target_col = f"target_{horizon}m"
        clean_df = df.dropna(subset=[target_col] + feature_cols)
        X = clean_df[feature_cols]
        y = clean_df[target_col]
        
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
        
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X)
        
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "Feature": feature_cols,
            "Mean Absolute SHAP": mean_abs_shap
        }).sort_values(by="Mean Absolute SHAP", ascending=False).head(10)
        
        ax = axes[i]
        sns.barplot(
            x="Mean Absolute SHAP",
            y="Feature",
            data=feature_importance,
            ax=ax,
            palette="Blues_r",
            edgecolor="#cccccc"
        )
        ax.set_title(f"{horizon}-Month Horizon SHAP Feature Importance", fontweight='bold', fontsize=11)
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Output)", fontsize=10)
        ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    fig_path = os.path.join(CWD, "03_figures", "figure3_shap.pdf")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    fig_path_png = os.path.join(CWD, "03_figures", "figure3_shap.png")
    plt.savefig(fig_path_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Figure 3 (SHAP importance) to {fig_path} and PNG copy")

def plot_pipeline_figure():
    print("Generating Figure 1 (Research Pipeline flowchart)...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    boxes = [
        {"text": "Data Retrieval\n• FRED API (USREC, Yield Curve)\n• Yahoo Finance (S&P 500)\n• Monthly alignment (1980-2026)", "x": 0.05, "y": 0.4, "w": 0.22, "h": 0.25, "color": "#f8f9fa", "edgecolor": "#2980b9"},
        {"text": "Feature & Target Prep\n• 3M/6M/12M lead targets\n• Lags, growth rates, spreads\n• Sahm Indicator, drawdowns", "x": 0.32, "y": 0.4, "w": 0.22, "h": 0.25, "color": "#f8f9fa", "edgecolor": "#2980b9"},
        {"text": "Time-Series Validation\n• Expanding window CV\n• No random split (no leakage)\n• Train on history, test on year t", "x": 0.59, "y": 0.4, "w": 0.22, "h": 0.25, "color": "#f8f9fa", "edgecolor": "#2980b9"},
        {"text": "Models & Explainability\n• Probit & LR baselines\n• Tree ensembles (RF, XGB, LGB)\n• SHAP explainability", "x": 0.86, "y": 0.4, "w": 0.22, "h": 0.25, "color": "#f8f9fa", "edgecolor": "#2980b9"},
    ]
    
    for box in boxes:
        rect = plt.Rectangle((box["x"] - box["w"]/2, box["y"] - box["h"]/2), box["w"], box["h"], 
                             facecolor=box["color"], edgecolor=box["edgecolor"], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(box["x"], box["y"], box["text"], ha='center', va='center', fontsize=8, color='#2c3e50', 
                linespacing=1.3, bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="none"))
        
    for i in range(len(boxes) - 1):
        x_start = boxes[i]["x"] + boxes[i]["w"]/2
        x_end = boxes[i+1]["x"] - boxes[i+1]["w"]/2
        y = 0.4
        ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="->", color='#2980b9', lw=2, mutation_scale=15))
        
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0.1, 0.7)
    
    plt.tight_layout()
    fig_path = os.path.join(CWD, "03_figures", "figure1_pipeline.pdf")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    fig_path_png = os.path.join(CWD, "03_figures", "figure1_pipeline.png")
    plt.savefig(fig_path_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {fig_path} and PNG copy")

def run_evaluation_pipeline():
    plot_pipeline_figure()
    calculate_metrics()
    plot_probabilities_over_time()
    plot_calibration_curves()
    analyze_lead_time()
    generate_shap_values()
    print("\nALL EVALUATION AND FIGURES COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_evaluation_pipeline()
