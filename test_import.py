#!/usr/bin/env python3
"""
Simple test script to verify that the package can be imported correctly
"""

def test_imports():
    """Test importing main components of the package"""
    import crypto_backtester
    print(f"Successfully imported crypto_backtester v{crypto_backtester.__version__}")
    
    # Try importing main components
    from crypto_backtester import BacktestingEnv, calculate_performance_metrics, validate_config
    print("Successfully imported BacktestingEnv, calculate_performance_metrics, validate_config")
    
    # Try importing submodules
    from crypto_backtester.data import fetch_binance_data
    from crypto_backtester.indicators import add_indicators
    from crypto_backtester.metrics import calculate_sortino_ratio, calculate_sharpe_ratio
    from crypto_backtester.utils import estimate_periods_per_year
    
    print("Successfully imported all submodules")
    
    # Create an empty config for validation
    test_config = {}
    validated_config = validate_config(test_config)
    print(f"Config validation works, default symbol: {validated_config['symbol']}")
    
    print("All imports successful!")

if __name__ == "__main__":
    test_imports() 