"""
Fortress - Cryptocurrency Trading and Backtesting Framework
"""

__version__ = "0.1.0"

# Expose main components at the top level
from fortress.environment.trading_env import BacktestingEnv
from fortress.metrics.performance import calculate_performance_metrics
from fortress.utils.helpers import validate_config

__all__ = ['BacktestingEnv', 'calculate_performance_metrics', 'validate_config']
