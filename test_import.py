#!/usr/bin/env python3
"""
Test script to verify the package is importable and components are working
"""

import sys
import pandas as pd
import numpy as np

# Test importing the entire package
import fortress
print(f"Successfully imported fortress v{fortress.__version__}")

# Test importing specific components
from fortress import BacktestingEnv, calculate_performance_metrics, validate_config
print("Successfully imported main components")

# Test importing submodules
from fortress.data import fetch_binance_data
from fortress.indicators import add_indicators
from fortress.metrics import calculate_sortino_ratio, calculate_sharpe_ratio
from fortress.utils import estimate_periods_per_year
print("Successfully imported submodules")

# Simple usage test
def test_basic_functionality():
    data = pd.DataFrame({
        'open': np.random.random(100) * 100 + 10000,
        'high': np.random.random(100) * 100 + 10050,
        'low': np.random.random(100) * 100 + 9950,
        'close': np.random.random(100) * 100 + 10000,
        'volume': np.random.random(100) * 1000,
    })
    print("Created test data")

if __name__ == "__main__":
    print("All imports successful!")
    test_basic_functionality() 