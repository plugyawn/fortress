"""
Simple example of using the crypto_backtester library
"""

import time
import logging
from binance.client import Client

from crypto_backtester import BacktestingEnv, calculate_performance_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_simple_strategy():
    """Runs a very simple strategy to demonstrate the library."""
    print("--- Starting Simple Crypto Backtesting Example ---")

    # --- Environment Setup ---
    env_config = {
        "symbol": "ETHUSDT",
        "interval": Client.KLINE_INTERVAL_4HOUR,  # 4-hour timeframe
        "start_date": "2023-01-01",
        "end_date": "2023-09-30",
        "lookback_window": 30,
        "sortino_period": 60,
        "render_mode": 'human'  # 'human' for detailed prints, 'none' for silent
    }

    env = BacktestingEnv(config=env_config)

    # --- Run Backtest ---
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward_accumulated = 0.0
    step_count = 0

    while not terminated and not truncated:
        step_count += 1
        
        # Get current state summary
        state_summary = env.get_state_summary(info)
        print(f"\n--- Step {info['step']} State ---")
        print(state_summary)

        # Simple strategy logic:
        # - Buy if we don't have a position and rolling Sortino is positive
        # - Sell if unrealized loss > 2% or profit > 5%
        action = 0  # Default: Hold
        
        if info['position_size'] < 1e-8:  # If flat (no position)
            if info['rolling_sortino'] > 0.1:
                action = 1  # Buy
                strategy_reasoning = "Strategy decides to BUY (Rolling Sortino positive)"
            else:
                strategy_reasoning = "Strategy decides to HOLD (Flat, Sortino not compelling)"
        else:  # If long (have position)
            if info['unrealized_pnl_perc'] < -0.02:
                action = 2  # Sell (Stop Loss)
                strategy_reasoning = "Strategy decides to SELL (Stop Loss Triggered)"
            elif info['unrealized_pnl_perc'] > 0.05:
                action = 2  # Sell (Take Profit)
                strategy_reasoning = "Strategy decides to SELL (Take Profit Triggered)"
            else:
                strategy_reasoning = "Strategy decides to HOLD (Position Open)"

        print(f">>> Strategy: {strategy_reasoning}")
        print(f">>> Action: {action} ({ {0: 'Hold', 1: 'Buy', 2: 'Sell'}[action] })")

        # Execute action in environment
        try:
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward_accumulated += reward
        except Exception as e:
            logger.error(f"Error during env.step(): {e}", exc_info=True)
            terminated = True  # Stop simulation on error

        # Small delay for human reading if rendering
        if env.render_mode == 'human':
            time.sleep(0.1)

    # --- Calculate and Display Performance ---
    print("\n--- Backtest Finished ---")

    performance_metrics = calculate_performance_metrics(
        portfolio_history=env.portfolio_history,
        trade_history=env.trade_history,
        initial_capital=env.initial_capital,
        risk_free_rate_annual=env.config['reward_risk_free_rate'],
        periods_per_year=env._periods_per_year,
        total_steps=env.current_step
    )

    print("\n--- Performance Metrics ---")
    if "Error" in performance_metrics or "Warning" in performance_metrics:
        print(performance_metrics)
    else:
        for key, value in performance_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")

    print(f"\nTotal accumulated reward (Sum of Diff Sortino): {total_reward_accumulated:.6f}")

    env.close()
    print("\n--- Example Run Complete ---")


if __name__ == "__main__":
    run_simple_strategy() 