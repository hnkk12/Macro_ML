import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"

MODEL_NAMES = {
    'prob_probit': 'Probit Baseline',
    'prob_lr': 'Logistic Regression',
    'prob_rf': 'Random Forest',
    'prob_xgb': 'XGBoost',
    'prob_lgb': 'LightGBM',
    'prob_stack': 'Stacked Ensemble'
}

MODEL_COLORS = {
    'prob_probit': '#7f8c8d',     # Slate gray
    'prob_lr': '#34495e',         # Dark navy
    'prob_rf': '#e67e22',         # Orange
    'prob_xgb': '#1abc9c',         # Turquoise
    'prob_lgb': '#2980b9',         # Light blue
    'prob_stack': '#8e44ad'        # Purple (Premium color for Stacking)
}

# Premium plotting stylesheet
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
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

def load_predictions(horizon):
    path = os.path.join(CWD, "models", f"predictions_{horizon}m.csv")
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    return df

def calculate_metrics():
    print("[Evaluation] Calculating performance metrics...")
    results = []
    
    for horizon in [3, 6, 12]:
        df = load_predictions(horizon).dropna(subset=["actual_target"])
        y_true = df["actual_target"]
        
        for col, model_name in MODEL_NAMES.items():
            y_prob = df[col]
            
            auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            
            # Find optimal threshold to maximize F1-score
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
    
    # Save CSV version
    tables_dir = os.path.join(CWD, "04_tables")
    os.makedirs(tables_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(tables_dir, "table2_performance.csv"), index=False)
    
    # Save formatted Excel table pivoted for presentation
    pivot_df = metrics_df.pivot(index="Model", columns="Horizon", values=["ROC-AUC", "PR-AUC", "F1-Score", "Brier Score"])
    pivoted_path = os.path.join(tables_dir, "table2_performance.xlsx")
    pivot_df.to_excel(pivoted_path)
    print(f"[Evaluation] Saved Table 2 (performance) to {pivoted_path}")
    
    return metrics_df

def plot_probabilities_over_time():
    print("[Evaluation] Plotting recession probabilities over time...")
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
            # Show Probit, XGBoost, and the Stacking Ensemble for clarity and contrast
            if col in ['prob_probit', 'prob_xgb', 'prob_stack']:
                ax.plot(df.index, df[col], color=MODEL_COLORS[col], linewidth=1.5, label=model_name)
                
        ax.set_title(f"Recession Risk Probability Forecast: {horizon}-Month Horizon", fontweight='bold', fontsize=11)
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    axes[2].set_xlabel("Year", fontsize=10)
    plt.tight_layout()
    
    fig_dir = os.path.join(CWD, "03_figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    png_path = os.path.join(fig_dir, "figure2_probabilities.png")
    pdf_path = os.path.join(fig_dir, "figure2_probabilities.pdf")
    
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Figure 2 to {png_path} and {pdf_path}")

def plot_calibration_curves():
    print("[Evaluation] Plotting calibration curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, horizon in enumerate([3, 6, 12]):
        df = load_predictions(horizon).dropna(subset=["actual_target"])
        y_true = df["actual_target"]
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
    fig_dir = os.path.join(CWD, "03_figures")
    
    png_path = os.path.join(fig_dir, "figure4_calibration.png")
    pdf_path = os.path.join(fig_dir, "figure4_calibration.pdf")
    
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Figure 4 to {png_path} and {pdf_path}")

def analyze_lead_time(threshold=0.25):
    print("[Evaluation] Analyzing warning lead times...")
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
                    if rec_start in probs.index and probs.loc[rec_start] >= threshold:
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
    lead_df.to_csv(os.path.join(tables_dir, "table3_leadtime.csv"), index=False)
    lead_df.to_excel(os.path.join(tables_dir, "table3_leadtime.xlsx"), index=False)
    print(f"[Evaluation] Saved Table 3 (lead times) to 04_tables/table3_leadtime.xlsx")
    
    return lead_df

def plot_pipeline_flowchart():
    print("[Evaluation] Generating Research Pipeline flowchart (Figure 1)...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    boxes = [
        {"text": "Data Retrieval\n• FRED API (Levels)\n• Yahoo Finance (S&P 500)\n• Monthly alignment (1980-2026)", "x": 0.05, "y": 0.4, "w": 0.22, "h": 0.25, "color": "#f8f9fa", "edgecolor": "#1abc9c"},
        {"text": "Feature & Target Prep\n• Lags, Vol, Momentum (~80 feats)\n• 3M/6M/12M lead targets\n• Sahm Indicator & Spreads", "x": 0.32, "y": 0.4, "w": 0.22, "h": 0.25, "color": "#f8f9fa", "edgecolor": "#2980b9"},
        {"text": "Time-Series Validation\n• Expanding-window year t\n• Tuning via TimeSeriesSplit\n• Strict temporal overlap gap", "x": 0.59, "y": 0.4, "w": 0.22, "h": 0.25, "color": "#f8f9fa", "edgecolor": "#8e44ad"},
        {"text": "Models & Explainability\n• Tuned RF, XGB, LGB, Stack\n• Probit & LR baselines\n• Orthogonal Category SHAP", "x": 0.86, "y": 0.4, "w": 0.22, "h": 0.25, "color": "#f8f9fa", "edgecolor": "#34495e"},
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
    fig_dir = os.path.join(CWD, "03_figures")
    
    png_path = os.path.join(fig_dir, "figure1_pipeline.png")
    pdf_path = os.path.join(fig_dir, "figure1_pipeline.pdf")
    
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Figure 1 to {png_path} and {pdf_path}")

def run_evaluation_pipeline():
    plot_pipeline_flowchart()
    calculate_metrics()
    plot_probabilities_over_time()
    plot_calibration_curves()
    analyze_lead_time()

if __name__ == "__main__":
    run_evaluation_pipeline()
