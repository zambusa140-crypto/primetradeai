#!/usr/bin/env python3
"""
End-to-end tests for MLOps Batch Job
Tests cover success paths, error handling, and determinism.
"""

import json
import os
import subprocess
import sys
import tempfile
import shutil


def run_pipeline(input_file, config_file, output_file, log_file):
    """Run the pipeline and return exit code."""
    cmd = [
        sys.executable, 'run.py',
        '--input', input_file,
        '--config', config_file,
        '--output', output_file,
        '--log-file', log_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def create_test_csv(path, rows=100):
    """Create a test CSV file with OHLCV data."""
    with open(path, 'w') as f:
        f.write('timestamp,open,high,low,close,volume\n')
        for i in range(rows):
            close = 100 + i * 0.5
            f.write(f'2024-01-01 {i:02d}:00:00,{close},{close+1},{close-1},{close},1000\n')


def create_test_config(path, seed=42, window=5, version='v1'):
    """Create a test config YAML file."""
    with open(path, 'w') as f:
        f.write(f'seed: {seed}\n')
        f.write(f'window: {window}\n')
        f.write(f'version: "{version}"\n')


def test_success_path():
    """Test successful pipeline execution."""
    print("Running test: success_path")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'data.csv')
        config_file = os.path.join(tmpdir, 'config.yaml')
        output_file = os.path.join(tmpdir, 'metrics.json')
        log_file = os.path.join(tmpdir, 'run.log')
        
        create_test_csv(input_file, rows=100)
        create_test_config(config_file)
        
        exit_code, stdout, stderr = run_pipeline(input_file, config_file, output_file, log_file)
        
        assert exit_code == 0, f"Expected exit code 0, got {exit_code}. stderr: {stderr}"
        
        with open(output_file, 'r') as f:
            metrics = json.load(f)
        
        assert metrics['status'] == 'success'
        assert metrics['rows_processed'] == 100
        assert metrics['metric'] == 'signal_rate'
        assert 'value' in metrics
        assert metrics['seed'] == 42
        assert metrics['version'] == 'v1'
        assert 'latency_ms' in metrics
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert 'Job started' in log_content
        assert 'Config loaded' in log_content
        assert 'Data loaded' in log_content
        assert 'status=success' in log_content
    
    print("  PASSED")


def test_determinism():
    """Test that multiple runs produce identical results."""
    print("Running test: determinism")
    
    results = []
    for _ in range(3):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'data.csv')
            config_file = os.path.join(tmpdir, 'config.yaml')
            output_file = os.path.join(tmpdir, 'metrics.json')
            log_file = os.path.join(tmpdir, 'run.log')
            
            create_test_csv(input_file, rows=100)
            create_test_config(config_file)
            
            run_pipeline(input_file, config_file, output_file, log_file)
            
            with open(output_file, 'r') as f:
                metrics = json.load(f)
            
            results.append(metrics['value'])
    
    assert len(set(results)) == 1, f"Results not deterministic: {results}"
    print("  PASSED")


def test_missing_input_file():
    """Test error handling for missing input file."""
    print("Running test: missing_input_file")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.yaml')
        output_file = os.path.join(tmpdir, 'metrics.json')
        log_file = os.path.join(tmpdir, 'run.log')
        
        create_test_config(config_file)
        
        exit_code, stdout, stderr = run_pipeline(
            '/nonexistent/file.csv', config_file, output_file, log_file
        )
        
        assert exit_code != 0, "Expected non-zero exit code for missing input"
        
        with open(output_file, 'r') as f:
            metrics = json.load(f)
        
        assert metrics['status'] == 'error'
        assert 'error_message' in metrics
    
    print("  PASSED")


def test_missing_config_file():
    """Test error handling for missing config file."""
    print("Running test: missing_config_file")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'data.csv')
        output_file = os.path.join(tmpdir, 'metrics.json')
        log_file = os.path.join(tmpdir, 'run.log')
        
        create_test_csv(input_file)
        
        exit_code, stdout, stderr = run_pipeline(
            input_file, '/nonexistent/config.yaml', output_file, log_file
        )
        
        assert exit_code != 0, "Expected non-zero exit code for missing config"
    
    print("  PASSED")


def test_empty_csv():
    """Test error handling for empty CSV file."""
    print("Running test: empty_csv")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'data.csv')
        config_file = os.path.join(tmpdir, 'config.yaml')
        output_file = os.path.join(tmpdir, 'metrics.json')
        log_file = os.path.join(tmpdir, 'run.log')
        
        with open(input_file, 'w') as f:
            f.write('')
        
        create_test_config(config_file)
        
        exit_code, stdout, stderr = run_pipeline(input_file, config_file, output_file, log_file)
        
        assert exit_code != 0, "Expected non-zero exit code for empty CSV"
        
        with open(output_file, 'r') as f:
            metrics = json.load(f)
        
        assert metrics['status'] == 'error'
    
    print("  PASSED")


def test_missing_close_column():
    """Test error handling for missing 'close' column."""
    print("Running test: missing_close_column")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'data.csv')
        config_file = os.path.join(tmpdir, 'config.yaml')
        output_file = os.path.join(tmpdir, 'metrics.json')
        log_file = os.path.join(tmpdir, 'run.log')
        
        with open(input_file, 'w') as f:
            f.write('timestamp,open,high,low,volume\n')
            f.write('2024-01-01,100,101,99,1000\n')
        
        create_test_config(config_file)
        
        exit_code, stdout, stderr = run_pipeline(input_file, config_file, output_file, log_file)
        
        assert exit_code != 0, "Expected non-zero exit code for missing 'close' column"
        
        with open(output_file, 'r') as f:
            metrics = json.load(f)
        
        assert metrics['status'] == 'error'
        assert 'close' in metrics['error_message'].lower()
    
    print("  PASSED")


def test_invalid_config():
    """Test error handling for invalid config (missing required fields)."""
    print("Running test: invalid_config")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'data.csv')
        config_file = os.path.join(tmpdir, 'config.yaml')
        output_file = os.path.join(tmpdir, 'metrics.json')
        log_file = os.path.join(tmpdir, 'run.log')
        
        create_test_csv(input_file)
        
        with open(config_file, 'w') as f:
            f.write('seed: 42\n')
        
        exit_code, stdout, stderr = run_pipeline(input_file, config_file, output_file, log_file)
        
        assert exit_code != 0, "Expected non-zero exit code for invalid config"
    
    print("  PASSED")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Running E2E Tests for MLOps Batch Job")
    print("=" * 50)
    
    tests = [
        test_success_path,
        test_determinism,
        test_missing_input_file,
        test_missing_config_file,
        test_empty_csv,
        test_missing_close_column,
        test_invalid_config,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
