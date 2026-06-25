import sys
import os

CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"
sys.path.append(CWD)

from src.features import run_feature_pipeline
from src.models import run_walk_forward_backtest
from src.evaluation import run_evaluation_pipeline
from src.explain import generate_shap_plots

def run_main_pipeline():
    """
    Executes the entire research pipeline sequentially.
    """
    print("==================================================================")
    print("STARTING EXPLAINABLE MULTI-HORIZON RECESSION FORECASTING PIPELINE")
    print("==================================================================")
    
    # 1. Feature Engineering & Target Prep
    print("\n[Step 1/4] Running Feature Engineering and Target Construction...")
    run_feature_pipeline()
    
    # 2. Expanding-Window Time-Series Cross-Validation
    print("\n[Step 2/4] Running walk-forward expanding window cross-validation...")
    run_walk_forward_backtest()
    
    # 3. Model Evaluation, Figures and Tables Generation
    print("\n[Step 3/4] Running evaluation, plotting probabilities and calibration curves...")
    run_evaluation_pipeline()
    
    # 4. Clean SHAP Explainability on Representative Categories
    print("\n[Step 4/4] Running SHAP explainability analysis on orthogonal categories...")
    generate_shap_plots()
    
    print("\n==================================================================")
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("==================================================================")

if __name__ == "__main__":
    run_main_pipeline()
