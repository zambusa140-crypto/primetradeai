# MLOps Batch Job - Trading Signal Pipeline

This project implements a minimal MLOps-style batch job that demonstrates reproducibility, observability, and deployment readiness.

## Overview

The pipeline:
1. Loads configuration from YAML
2. Reads OHLCV data from CSV
3. Computes rolling mean on close prices
4. Generates binary trading signals (1 if close > rolling mean, else 0)
5. Outputs structured metrics JSON and detailed logs

## Local Run Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## Docker Build/Run Commands

```bash
# Build the Docker image
docker build -t mlops-task .

# Run the container (produces metrics.json and run.log)
docker run --rm mlops-task
```

## Configuration (config.yaml)

- `seed`: Random seed for reproducibility
- `window`: Rolling window size for mean calculation
- `version`: Pipeline version identifier

## Output Files

### metrics.json
Contains processing metrics including:
- `version`: Pipeline version
- `rows_processed`: Number of rows processed
- `metric`: Name of the computed metric
- `value`: Signal rate (mean of signals)
- `latency_ms`: Total runtime in milliseconds
- `seed`: Random seed used
- `status`: "success" or "error"

### run.log
Detailed log file with timestamps for:
- Job start/end
- Config validation
- Data loading
- Processing steps
- Metrics summary

## Example metrics.json

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4990,
  "latency_ms": 127,
  "seed": 42,
  "status": "success"
}
```

## Error Handling

The pipeline handles these error cases:
- Missing input file
- Invalid CSV format
- Empty file
- Missing required column (close)
- Invalid config structure

Error output includes the same structure with `status: "error"` and an `error_message` field.
