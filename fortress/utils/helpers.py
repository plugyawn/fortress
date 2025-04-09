"""
Utility helper functions for the backtesting framework
"""

from typing import Dict

def estimate_periods_per_year(interval: str) -> float:
    """Estimates the number of intervals in a year."""
    interval_map = {
        '1m': 60*24*365.25, '3m': 20*24*365.25, '5m': 12*24*365.25, '15m': 4*24*365.25,
        '30m': 2*24*365.25, '1h': 24*365.25, '2h': 12*365.25, '4h': 6*365.25,
        '6h': 4*365.25, '8h': 3*365.25, '12h': 2*365.25, '1d': 365.25, '1w': 52
    }
    if interval not in interval_map:
        return 365.25  # Default to daily
    return interval_map[interval]

def validate_config(config: Dict) -> Dict:
    """Validates and fills in defaults for configuration."""
    default_config = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 10000.0,
        "commission_perc": 0.001,  # 0.1%
        "slippage_perc": 0.0005,  # 0.05%
        "lookback_window": 50,
        "indicators_config": [
            {"kind": "rsi", "length": 14},
            {"kind": "macd"},
            {"kind": "bbands", "length": 20, "std": 2},
            {"kind": "atr", "length": 14},
            {"kind": "ema", "length": 50},
            {"kind": "ema", "length": 200},
            {"kind": "adx", "length": 14},
        ],
        "reward_risk_free_rate": 0.0,
        "reward_sortino_period": 120,
        "force_data_refresh": False,
        "cache_dir": "crypto_data_cache",
        "render_mode": 'human',
    }
    
    # Update with user-provided values
    validated_config = default_config.copy()
    validated_config.update({k: v for k, v in config.items() if k in default_config})
    
    return validated_config

def format_state_summary(state_info: Dict) -> str:
    """Formats environment state information into a readable summary."""
    summary = f"--- Trading Environment State (Step {state_info['step']}) ---\n"
    summary += f"Timestamp: {state_info['timestamp']}\n"
    summary += f"Portfolio Value: ${state_info['portfolio_value']:.2f}\n"
    summary += f"Cash: ${state_info['cash']:.2f}\n"

    if state_info['position_size'] > 1e-8:
        summary += f"Position: LONG {state_info['symbol']}\n"
        summary += f"  Size: {state_info['position_size']:.6f}\n"
        summary += f"  Entry Price: ${state_info['position_entry_price']:.2f}\n"
        summary += f"  Duration: {state_info['position_duration_steps']} steps\n"
        summary += f"  Current Market Price: ${state_info['market_price_close']:.2f}\n"
        summary += f"  Unrealized PnL: ${state_info['unrealized_pnl']:.2f} ({state_info['unrealized_pnl_perc']*100:.2f}%)\n"
    else:
        summary += "Position: FLAT (no open position)\n"

    summary += f"Risk Metrics (Rolling {state_info['sortino_period']}-step):\n"
    summary += f"  Sortino Ratio: {state_info['rolling_sortino']:.3f}\n"
    summary += f"  Sharpe Ratio: {state_info['rolling_sharpe']:.3f}\n"

    last_trade = state_info.get('last_trade')
    if last_trade:
        trade_type = last_trade['type']
        trade_price = last_trade['price']
        trade_step = last_trade['step']
        profit_str = ""
        if 'profit' in last_trade:
            profit_str = f", Profit: ${last_trade['profit']:.2f}"
        summary += f"Last Trade: {trade_type} at ${trade_price:.2f} (Step {trade_step}{profit_str})\n"

    summary += "--- End State ---"
    return summary 