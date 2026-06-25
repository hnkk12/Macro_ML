import os
import sys

CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"
if CWD not in sys.path:
    sys.path.append(CWD)

from src.pipeline import run_main_pipeline

if __name__ == "__main__":
    run_main_pipeline()
