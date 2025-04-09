"""
Crypto Backtester - A flexible cryptocurrency backtesting framework
"""

__version__ = "0.1.0"

# Import main components to make them available at the package level
from crypto_backtester.environment.trading_env import BacktestingEnv
from crypto_backtester.metrics.performance import calculate_performance_metrics
from crypto_backtester.utils.helpers import validate_config
