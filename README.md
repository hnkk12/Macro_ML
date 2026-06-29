# Research-Grade Recession Predictor Pipeline (v2.0.0)
## Target: CORE Rank B Conference Submission (PAKDD / ECML-PKDD / SDM / CIKM)

This repository contains a reproducible, leakage-free, and explainable Machine Learning pipeline for predicting macroeconomic recessions. The pipeline is designed to adhere to the rigorous standards of top-tier Data Mining and Machine Learning conferences, incorporating robust statistical tests and leakage-free validation frameworks.

---

## 1. Project Overview & Main Contributions

Traditional recession forecasting literature often relies on single economic models or suffers from subtle temporal data leakage. We address these drawbacks by introducing a research-ready pipeline with four main contributions:

*   **C1 — Hybrid Econometric-Machine Learning Stacking (HEML)**: We combine traditional Probit econometric models (providing structured parametric priors) with flexible machine learning models (Logistic Regression L2, Support Vector Machines (SVM), Multi-Layer Perceptrons (MLP), Random Forest, XGBoost, and LightGBM) via a stacked meta-learner. This bridges the gap between statistical rigor and machine learning flexibility.
*   **C2 — Leakage-free Expanding-Window Validation with Temporal Gap**: Many past studies suffer from data leakage when using forward-looking targets. Our splitter enforces a strict temporal gap equal to the forecast horizon $h$:
    $$\max(\text{train\_date}) \le \min(\text{test\_date}) - h$$
    This ensures that target windows of the training set do not overlap with the test period, eliminating optimistic bias.
*   **C3 — Regime-Specific Economic Interpretation**: Instead of reporting static global feature importances, we attribute risk factors locally to specific historical episodes (the 2001 Dot-com bubble, the 2008 Financial Crisis, and the 2020 Covid-19 pandemic) using SHAP values. This reveals that the driving factors vary dynamically across economic regimes.
*   **C4 — Pairwise Forecast Significance Testing (Diebold-Mariano) & Temporal Gap Sensitivity Analysis**: We implement a rigorous statistical validation framework. Using the Diebold-Mariano test, we verify whether our HEML framework significantly outperforms standard econometrics and ML baselines under temporal autocorrelation. Furthermore, we conduct a sensitivity analysis on temporal gap lengths to quantitatively demonstrate the impact of information leakage on predictive performance.

---

## 2. Macroeconomic Data Sources

We utilize 12 key macroeconomic and financial indicators from the Federal Reserve Economic Data (FRED) and Yahoo Finance:

| Series ID | Source | Description |
|-----------|--------|-------------|
| **USREC** | FRED | NBER Recession Indicator (target label: 1=recession, 0=expansion) |
| **PAYEMS** | FRED | Nonfarm Payrolls (employment momentum) |
| **UNRATE** | FRED | Civilian Unemployment Rate |
| **FEDFUNDS** | FRED | Effective Federal Funds Rate |
| **CPIAUCSL** | FRED | Consumer Price Index for All Urban Consumers (CPI) |
| **GS10** | FRED | 10-Year Treasury Constant Maturity Rate |
| **GS5** | FRED | 5-Year Treasury Constant Maturity Rate |
| **TB3MS** | FRED | 3-Month Treasury Bill Secondary Market Rate |
| **INDPRO** | FRED | Industrial Production Index |
| **T10Y2Y** | FRED | 10-Year Treasury minus 2-Year Treasury spread |
| **BAMLH0A0HYM2** | FRED | ICE BofA High Yield US High Yield Index Option-Adjusted Spread |
| **^GSPC** | Yahoo | S&P 500 Index (stock market returns and drawdown) |

---

## 3. Task Definition & Validation Setup

We model recession forecasting as a binary classification task. For a given forecast origin $t$ and forecast horizon $h \in \{3, 6, 12\}$ months, the target variable $y_t^{(h)}$ is defined as:
$$y_t^{(h)} = \mathbb{I} \left( \exists k \in [t+1, t+h] \text{ s.t. } \text{USREC}_{k} = 1 \right)$$
where $y_t^{(h)} = 1$ if a recession occurs at any month in the lookahead window, and $0$ otherwise.

### Expanding Window splits (Leakage-free)
To simulate out-of-sample forecasting, we run expanding window cross-validation starting with the training end date set to `2005-12-01` and out-of-sample testing from `2006-01-01` onwards.

```
Split 1:
Train: |===================| (ends 2005-12-01 - h months)
Gap:   |--- h months ---|
Test:                   |==== 12 months ====| (2006-01-01 to 2006-12-01)

Split 2:
Train: |=======================| (ends 2006-12-01 - h months)
Gap:   |--- h months ---|
Test:                       |==== 12 months ====| (2007-01-01 to 2007-12-01)
```

---

## 4. How to Run

Running the entire pipeline requires Python 3.10+ (tested on Python 3.12.2).

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Pipeline
You can execute the entire pipeline (downloading, building dataset, running validation, running robustness analysis, and creating tables & figures) sequentially:

```bash
# 1. Download raw datasets
python scripts/download_data.py --config configs/experiments/main.yaml

# 2. Preprocess and compile panel data
python scripts/build_dataset.py --config configs/experiments/main.yaml

# 3. Run primary out-of-sample experiments (model training & tuning)
python scripts/run_experiment.py --config configs/experiments/main.yaml

# 4. Run robustness gap sensitivity analysis (C4)
python scripts/run_robustness_gap.py

# 5. Generate tables (CSV outputs including DM test & robustness results)
python scripts/make_tables.py --config configs/experiments/main.yaml

# 6. Generate figures (PNG plots including weights dynamics & gap sensitivity)
python scripts/make_figures.py --config configs/experiments/main.yaml
```

---

## 5. Output Directory Structure

All generated tables, predictions, and charts are stored in the `outputs/` folder:

*   `outputs/predictions/`
    *   `oos_predictions_h3.csv`, `oos_predictions_h6.csv`, `oos_predictions_h12.csv`: Out-of-sample predictions.
*   `outputs/tables/`
    *   `table_main_metrics.csv`: ROC-AUC, PR-AUC, Brier score, ECE, Log-loss across models and horizons.
    *   `table_ablation.csv`: Performance comparison across feature sets (ablation study).
    *   `table_bootstrap_ci.csv`: Block bootstrap confidence intervals.
    *   `table_lead_time.csv`: Detection lead times per recession episode.
    *   `table_threshold_sensitivity.csv`: F1 score sensitivity to probability thresholds.
    *   `table_calibration.csv`: Expected Calibration Error comparisons.
    *   `table_significance_dm.csv`: Diebold-Mariano pairwise significance test results (C4).
    *   `table_robustness_gap.csv`: Temporal gap sensitivity analysis table (C4).
    *   `table_meta_weights.csv`: Meta-learner stacking weights dynamics (ablation).
*   `outputs/figures/`
    *   `probability_paths.png`: OOS probability paths versus NBER gray shading.
    *   `calibration_curves.png`: Reliability diagram for probabilitic calibration.
    *   `pr_curves.png`: Precision-Recall curves.
    *   `shap_by_episode.png`: Episode-specific mean absolute SHAP values (C3).
    *   `feature_stability.png`: Stability of standardized coefficients across split iterations.
    *   `meta_weights_dynamics.png`: Stacked area chart showing how model weights adapt dynamically across splits.
    *   `robustness_gap_sensitivity.png`: Performance decay curves showing the impact of temporal gap sizes.

---

## 6. Reproducibility & Testing

All random seeds are fixed globally via config (`seed: 42`). 

Run unit tests to verify target construction, temporal gap isolation, and scaling hygiene:
```bash
python -m pytest tests/ -v --tb=short
```

---

## 7. Robustness & Open Research Directions

1.  **US-Centric Evaluation**: In this phase, evaluation is limited to US macroeconomic indicators. Extending to G7 or OECD countries would improve generalization claims (parametric support is ready via `configs/experiments/main.yaml`).
2.  **Publication Lag**: The panel is built at a monthly frequency. Real-time implementation is subject to publication lag, which is partially mitigated by our `low_revision` feature set ablation.
3.  **Real-Time Vintage Data**: We use revised data. Future research could incorporate real-time vintage data to test performance under true historical information sets.
