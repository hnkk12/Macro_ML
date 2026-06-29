import argparse
import sys
import os

# Ensure package root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_experiment_config
from src.utils.logging_utils import get_logger
from src.data.downloader import run_download

logger = get_logger("download_data")

def main():
    parser = argparse.ArgumentParser(description="Download raw economic series from FRED and Yahoo.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML file.")
    args = parser.parse_args()
    
    logger.info(f"Loading config from {args.config}")
    config = load_experiment_config(args.config)
    
    logger.info("Starting data download...")
    run_download(config)
    logger.info("Data download process finished!")

if __name__ == "__main__":
    main()
