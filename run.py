#!/usr/bin/env python3
"""
MLOps Batch Job - Trading Signal Pipeline
Computes rolling mean on OHLCV data and generates binary signals.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path, logger):
    """Load and validate configuration from YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_fields = ['seed', 'window', 'version']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    logger.info(f"Config loaded - seed={config['seed']}, window={config['window']}, version={config['version']}")
    return config


def load_data(input_path, logger):
    """Load and validate input CSV data."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Invalid CSV format: {e}")
    
    if df.empty:
        raise ValueError("Input file is empty")
    
    if 'close' not in df.columns:
        raise ValueError("Missing required column: 'close'")
    
    logger.info(f"Data loaded - {len(df)} rows")
    return df


def compute_rolling_mean(close_series, window):
    """Compute rolling mean, returning NaN for first (window-1) rows."""
    return close_series.rolling(window=window).mean()


def generate_signal(close_series, rolling_mean):
    """Generate binary signal: 1 if close > rolling_mean, else 0."""
    return (close_series > rolling_mean).astype(int)


def write_metrics(output_path, metrics):
    """Write metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def write_error_metrics(output_path, version, error_message):
    """Write error metrics to JSON file."""
    metrics = {
        "version": version,
        "status": "error",
        "error_message": error_message
    }
    write_metrics(output_path, metrics)


def main():
    parser = argparse.ArgumentParser(description='MLOps Batch Job')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--config', required=True, help='Config YAML file path')
    parser.add_argument('--output', required=True, help='Output metrics JSON file path')
    parser.add_argument('--log-file', required=True, help='Log file path')
    
    args = parser.parse_args()
    
    start_time = time.time()
    logger = setup_logging(args.log_file)
    
    logger.info(f"Job started at {datetime.now().isoformat()}")
    
    version = "unknown"
    
    try:
        # Load and validate config
        config = load_config(args.config, logger)
        version = config['version']
        
        # Set random seed for reproducibility
        np.random.seed(config['seed'])
        
        # Load and validate data
        df = load_data(args.input, logger)
        
        # Compute rolling mean
        logger.info("Computing rolling mean")
        df['rolling_mean'] = compute_rolling_mean(df['close'], config['window'])
        
        # Generate signal (NaN rows will get 0 since comparison with NaN is False)
        logger.info("Generating signals")
        df['signal'] = generate_signal(df['close'], df['rolling_mean'])
        
        # Calculate metrics
        rows_processed = len(df)
        # Only count non-NaN signals for signal rate
        valid_signals = df['signal'].dropna()
        signal_rate = float(valid_signals.mean()) if len(valid_signals) > 0 else 0.0
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": config['seed'],
            "status": "success"
        }
        
        logger.info(f"Metrics summary - rows={rows_processed}, signal_rate={signal_rate:.4f}, latency={latency_ms}ms")
        
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        latency_ms = int((time.time() - start_time) * 1000)
        write_error_metrics(args.output, version, str(e))
        logger.info(f"Job ended at {datetime.now().isoformat()} - status=error")
        sys.exit(1)
    
    # Write success metrics
    write_metrics(args.output, metrics)
    
    logger.info(f"Job ended at {datetime.now().isoformat()} - status=success")
    
    # Print final metrics to stdout
    print(json.dumps(metrics, indent=2))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
