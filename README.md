[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-go.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)


<div align = center>
<a href = "github.com/plugyawn"><img width="800px" src= "https://github.com/user-attachments/assets/bd41262d-7919-47eb-9013-c161968a4410"></a>
</div>

-----------------------------------------
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)![Compatibility](https://img.shields.io/badge/compatible%20with-python3.6.x-blue.svg)

# Crypto Backtester

A high-performance cryptocurrency backtesting framework with realistic execution modeling. Designed for quantitative traders, researchers, and machine learning practitioners who need accurate and flexible backtesting capabilities.

## 🚀 Features

- **Data Integration**: Seamless Binance data fetching with intelligent caching
- **Gymnasium Compatible**: Follows OpenAI's Gym interface for easy integration with RL agents
- **Realistic Execution**: Models real-world trading with next candle open price and configurable slippage
- **Rich Observation Space**: Enhanced observation with normalized price data, indicators, and time features
- **Advanced Reward Shaping**: Uses differential Sortino ratio for optimizing risk-adjusted returns
- **Comprehensive Metrics**: Track Sharpe, Sortino, drawdowns, win rates, and more
- **Explanatory Summaries**: Generate human-readable state descriptions for debugging or LLM integration
- **Configurable Technical Indicators**: Easily add custom indicators via pandas_ta
- **CLI & API Support**: Use as a command-line tool or integrate into your Python projects

## 📋 Installation

### Prerequisites

- Python 3.7+
- Binance API keys (read-only access is sufficient for backtesting)

### From PyPI

```bash
pip install crypto-backtester
```

### From Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-backtester.git
cd crypto-backtester

# Install the package in development mode
pip install -e .
```

### Environment Variables

Set your Binance API keys (read-only is sufficient for fetching data):

```bash
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_API_SECRET="your_binance_api_secret"
```

## 🎮 Usage

### Command Line Interface

Run a quick backtest with the built-in HOLD strategy:

```bash
# Simple HOLD strategy backtest
crypto-backtest --symbol BTCUSDT --interval 1h --start_date 2022-01-01 --end_date 2022-12-31 --capital 10000

# Run in quiet mode (no step-by-step output)
crypto-backtest --symbol ETHUSDT --interval 4h --start_date 2023-01-01 --end_date 2023-06-30 --quiet

# Use a custom config file
crypto-backtest --config my_backtest_config.json
```

### Python API

```python
from crypto_backtester import BacktestingEnv, calculate_performance_metrics

# Configure the environment
config = {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "initial_capital": 10000.0,
    "commission_perc": 0.001,  # 0.1%
}

# Create and run the environment
env = BacktestingEnv(config)
observation, info = env.reset()

# Simple trading loop
for _ in range(100):
    # Replace this with your strategy logic
    action = 0  # 0: Hold, 1: Buy, 2: Sell
    
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Portfolio value: ${info['portfolio_value']:.2f}")
    
    if terminated or truncated:
        break

# Calculate performance metrics
metrics = calculate_performance_metrics(
    portfolio_history=env.portfolio_history,
    trade_history=env.trade_history,
    initial_capital=env.initial_capital,
    risk_free_rate_annual=0.0,
    periods_per_year=env._periods_per_year,
    total_steps=env.current_step
)
print(metrics)
```

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `symbol` | Trading pair | `"BTCUSDT"` |
| `interval` | Candle timeframe | `"1h"` |
| `start_date` | Backtest start date | `"2022-01-01"` |
| `end_date` | Backtest end date | `"2023-12-31"` |
| `initial_capital` | Starting capital | `10000.0` |
| `commission_perc` | Trading commission % | `0.001` (0.1%) |
| `slippage_perc` | Execution slippage % | `0.0005` (0.05%) |
| `lookback_window` | Historical data points in observation | `50` |
| `indicators_config` | Technical indicators to calculate | See example below |
| `reward_risk_free_rate` | Risk-free rate for Sortino calculation | `0.0` |
| `reward_sortino_period` | Lookback period for rolling Sortino | `120` |
| `force_data_refresh` | Skip cache and fetch new data | `false` |
| `cache_dir` | Directory for cached data | `"crypto_data_cache"` |
| `render_mode` | Rendering mode | `"human"` |

### Example Indicators Configuration

```python
indicators_config = [
    {"kind": "rsi", "length": 14},
    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
    {"kind": "bbands", "length": 20, "std": 2},
    {"kind": "ema", "length": 50},
    {"kind": "ema", "length": 200},
    {"kind": "adx", "length": 14}
]
```

## 📈 Example Strategies

### Simple Moving Average Crossover

```python
def ma_crossover_strategy(info, observation):
    """Simple moving average crossover strategy."""
    # Access indicators in the observation
    fast_ema = info.get('ema_50', 0)  # 50-period EMA
    slow_ema = info.get('ema_200', 0)  # 200-period EMA
    
    # Buy when fast crosses above slow
    if fast_ema > slow_ema and info['position_size'] < 1e-8:
        return 1  # Buy
    
    # Sell when fast crosses below slow
    elif fast_ema < slow_ema and info['position_size'] > 1e-8:
        return 2  # Sell
    
    return 0  # Hold
```

### RSI Mean Reversion

```python
def rsi_strategy(info, observation):
    """RSI mean reversion strategy."""
    rsi = info.get('rsi_14', 50)  # 14-period RSI
    
    # Buy when oversold
    if rsi < 30 and info['position_size'] < 1e-8:
        return 1  # Buy
    
    # Sell when overbought
    elif rsi > 70 and info['position_size'] > 1e-8:
        return 2  # Sell
    
    return 0  # Hold
```

## 🏗️ Architecture

The framework is organized into modular components:

- **data**: Handles data fetching and preprocessing
- **indicators**: Calculates technical indicators
- **environment**: The core backtesting environment (Gym-compatible)
- **metrics**: Performance calculation utilities
- **utils**: Helper functions and configuration tools
- **examples**: Example strategies and usage patterns
- **cli**: Command-line interface

## 📊 Performance Metrics

The backtester calculates comprehensive performance metrics:

- Total Return (%)
- Maximum Drawdown (%)
- Sharpe Ratio
- Sortino Ratio
- Win Rate (%)
- Profit Factor
- Average Win/Loss
- Total Commissions

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/crypto-backtester.git
cd crypto-backtester

# Install in development mode with testing dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical indicators
- [Gymnasium](https://gymnasium.farama.org/) for environment interface
- [python-binance](https://github.com/sammchardy/python-binance) for exchange API integration
