import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.utils.logging_utils import get_logger

logger = get_logger("tables")

def save_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV, creating parent directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved table to {filepath}")

def generate_target_summary(df: pd.DataFrame, horizons: List[int], output_path: str) -> None:
    """
    Generate target summary table (recession/expansion months count per horizon).
    """
    rows = []
    for h in horizons:
        col = f"Recession_within_{h}mo"
        if col in df.columns:
            y = df[col].dropna()
            total = len(y)
            pos = int(y.sum())
            neg = total - pos
            pos_pct = (pos / total) * 100 if total > 0 else 0
            rows.append({
                "horizon": h,
                "total_months": total,
                "recession_months": pos,
                "expansion_months": neg,
                "recession_pct": round(pos_pct, 2)
            })
            
    summary_df = pd.DataFrame(rows)
    save_csv(summary_df, output_path)
