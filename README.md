# Explainable Multi-Horizon U.S. Recession Risk Forecasting

> **ACML 2026 Conference Track -- Double-Blind Review Submission**
> 
> *This repository contains the replication code, processed datasets, figures, and tables for the research project. All author names and affiliations have been omitted to maintain anonymity.*

---

## 📈 Project Flow & Pipeline

The project follows a modular, structured pipeline designed to prevent temporal look-ahead bias and information leakage. The end-to-end execution flow is as follows:

```
[Raw Data Downloader] (01_download_data)
        │
        ▼
[Feature & Target Prep] (02_preprocess_targets) ──> Writes Table 1 (Variables)
        │
        ▼
[Model Training & CV] (03_models) ──────────────> Generates Out-of-Sample Predictions
        │
        ▼
[Evaluation & SHAP] (04_evaluation_shap) ────────> Generates Table 2 (Metrics),
                                                   Table 3 (Lead Times),
                                                   and Figures 1-4 (Plots)
```

1. **Data Retrieval**: Automated download of macroeconomic indicators from FRED API and stock indices from Yahoo Finance.
2. **Feature Engineering & Target Construction**: Alignment of series to a monthly frequency, calculation of lags and growth rates, implementation of the Sahm Rule, and building multi-horizon future targets (3M, 6M, and 12M).
3. **Time-Series Cross-Validation**: Walk-forward expanding-window validation starting from 2006 to 2026. Econometric baselines (Probit, Logistic Regression) and machine learning ensembles (Random Forest, XGBoost, LightGBM) are trained on historical data and tested year-by-year.
4. **Metrics & Interpretability**: Quantitative evaluation using ROC-AUC, PR-AUC, F1, and Brier Score. Early-warning lead time and probability calibration check. Features are explained using SHAP values.

---

## 📂 Repository Structure

The code and data package is organized as follows:

* **`01_data/`**: Datasets and data dictionary.
  * `raw_fred_data.csv`: Raw aggregated monthly data.
  * `processed_monthly_dataset.csv`: Engineered feature matrix and aligned multi-horizon targets.
  * `data_dictionary.csv`: Source and transformation descriptions.
* **`02_code/`**: Executable Jupyter Notebooks corresponding to the pipeline steps.
  * `01_download_data.ipynb`: Step 1 - Downloader.
  * `02_preprocess_targets.ipynb`: Step 2 - Preprocessor.
  * `03_models.ipynb`: Step 3 - Walk-forward model training.
  * `04_evaluation_shap.ipynb`: Step 4 - Analysis, plotting, and explainability.
* **`03_figures/`**: Generated academic figures (saved in both PDF and PNG formats).
  * `figure1_pipeline.pdf` / `.png`: Research pipeline schematic.
  * `figure2_probabilities.pdf` / `.png`: Recession probabilities vs shaded actual NBER recessions.
  * `figure3_shap.pdf` / `.png`: SHAP feature importance analysis (3M, 6M, 12M).
  * `figure4_calibration.pdf` / `.png`: Probability calibration curves.
* **`04_tables/`**: Excel spreadsheets of metrics and variables.
  * `table1_variables.xlsx`: Complete variable details (Table 1).
  * `table2_performance.xlsx`: Quantitative model evaluations (Table 2).
  * `table3_leadtime.xlsx`: Lead time triggers before recession start (Table 3).
* **`models/`**: Stores prediction files generated during CV.
  * `predictions_3m.csv`, `predictions_6m.csv`, `predictions_12m.csv`.
* **`06_submission_evidence/`**: Verification requirements.
  * `README.md`: Guidelines for venue and submission confirmations.

---

## 🚀 Environment Setup & Execution Instructions

To replicate the study findings and run the pipeline, follow these steps in your terminal:

### 1. Prerequisites & Environment Setup
Make sure you have Python 3.12 (or a compatible version) installed. Install all the required packages using the `requirements.txt` file:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows

# Install required dependencies
pip install -r requirements.txt
```

*Note: Ensure you have Jupyter or a Jupyter-compatible IDE (like VS Code or JupyterLab) installed to run the `.ipynb` files.*

### 2. Running the Pipeline
Run the notebooks in the `02_code/` directory sequentially. Each step outputs its results to the relevant data, figures, or tables folder.

#### Step 1: Download Raw Data
Open and execute all cells in **`02_code/01_download_data.ipynb`**. 
- **What it does**: Queries the FRED API for the 10 macroeconomic series and Yahoo Finance for the S&P 500 historical price. It cleans, aggregates, and aligns all observations to a unified monthly frequency.
- **Output**: Generates `01_data/raw_fred_data.csv`.

#### Step 2: Feature Engineering & Target Preparation
Open and execute all cells in **`02_code/02_preprocess_targets.ipynb`**.
- **What it does**: Aligns the 3-month, 6-month, and 12-month future-looking recession targets. Computes levels, differences, and logs of features, evaluates the Sahm Rule indicator, and builds lags (1, 3, 6, 12 months) of yield and credit spreads.
- **Output**: Generates `01_data/processed_monthly_dataset.csv` and the data dictionary in `04_tables/table1_variables.xlsx`.

#### Step 3: Run expanding-window CV
Open and execute all cells in **`02_code/03_models.ipynb`**.
- **What it does**: Executes walk-forward expanding-window cross-validation from 2006 to 2026. For each test year, it fits five models (Probit, Logistic Regression, Random Forest, XGBoost, and LightGBM) on all past data (removing overlap to prevent leakage) and outputs out-of-sample probability risk scores.
- **Output**: Saves prediction files to the `models/` directory (e.g. `predictions_3m.csv`).

#### Step 4: Evaluate Metrics, Calibration, and SHAP
Open and execute all cells in **`02_code/04_evaluation_shap.ipynb`**.
- **What it does**: Evaluates predictive performance (ROC-AUC, PR-AUC, F1-score, Brier score), calculates lead times prior to recession onset, plots probability calibration curves, and retrains the model to generate SHAP feature explanations.
- **Output**: Generates all tables in `04_tables/` and figures in `03_figures/`.
