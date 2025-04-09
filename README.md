# Crypto Backtester

A flexible cryptocurrency backtesting framework with realistic execution modeling.

## Features

- Data fetching from Binance with caching
- Gymnasium environment interface (reset, step)
- Realistic execution (next open + slippage)
- Rich context in observation space
- Enhanced observation space (time features, percentage change normalization)
- Advanced reward shaping (differential Sortino ratio)
- Detailed performance metrics (Sharpe, Sortino, Drawdown)
- State summarization for prompts
- Configurable indicators via pandas_ta

## Installation

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

## Usage

### Command Line

```bash
# Run a simple HOLD strategy backtest
crypto-backtest --symbol BTCUSDT --interval 1h --start_date 2022-01-01 --end_date 2022-12-31 --capital 10000

# Run in quiet mode (no step-by-step output)
crypto-backtest --symbol ETHUSDT --interval 4h --start_date 2023-01-01 --end_date 2023-06-30 --quiet

# Use a config file
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

for _ in range(100):
    action = 0  # 0: Hold, 1: Buy, 2: Sell
    observation, reward, terminated, truncated, info = env.step(action)
    
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

## Custom Strategies

To implement custom strategies, you can:

1. Create a strategy function that decides on actions based on environment state
2. Use the environment's step method to execute actions
3. Calculate and analyze performance metrics

See `crypto_backtester/examples/simple_backtest.py` for a complete example.

## License

MIT 