"""
Command line interface for the crypto_backtester package
"""

import argparse
import json
import logging
from binance.client import Client

from crypto_backtester import BacktestingEnv, calculate_performance_metrics

def main():
    """Main entry point for the CLI"""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crypto Backtester CLI')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading symbol (e.g. BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h',
                        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'],
                        help='Timeframe interval')
    parser.add_argument('--start_date', type=str, default='2022-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000.0,
                        help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate (as decimal, e.g. 0.001 for 0.1%%)')
    parser.add_argument('--slippage', type=float, default=0.0005,
                        help='Slippage rate (as decimal, e.g. 0.0005 for 0.05%%)')
    parser.add_argument('--lookback', type=int, default=50,
                        help='Lookback window for observations')
    parser.add_argument('--quiet', action='store_true',
                        help='Run in quiet mode (no step-by-step output)')
    parser.add_argument('--config', type=str,
                        help='Path to JSON config file (overrides other arguments)')
                        
    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {args.config}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return
    else:
        # Create config from command line arguments
        config = {
            "symbol": args.symbol,
            "interval": getattr(Client, f"KLINE_INTERVAL_{args.interval.upper()}"),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "initial_capital": args.capital,
            "commission_perc": args.commission,
            "slippage_perc": args.slippage,
            "lookback_window": args.lookback,
            "render_mode": 'none' if args.quiet else 'human',
            # Use default values for other settings
            "reward_risk_free_rate": 0.0,
            "reward_sortino_period": 60,
            "force_data_refresh": False,
            "cache_dir": "crypto_data_cache",
        }

    # Run backtest with simple hold strategy
    logger.info(f"Starting backtest for {config['symbol']} from {config['start_date']} to {config['end_date']}")
    
    try:
        env = BacktestingEnv(config=config)
        observation, info = env.reset()
        
        terminated = False
        truncated = False
        total_steps = 0
        
        # Simple "hold" strategy
        while not terminated and not truncated:
            # Always hold (action=0)
            observation, reward, terminated, truncated, info = env.step(0)
            total_steps += 1
            
            if total_steps % 100 == 0:
                logger.info(f"Completed {total_steps} steps...")
        
        # Calculate and display performance metrics
        metrics = calculate_performance_metrics(
            portfolio_history=env.portfolio_history,
            trade_history=env.trade_history,
            initial_capital=env.initial_capital,
            risk_free_rate_annual=env.config['reward_risk_free_rate'],
            periods_per_year=env._periods_per_year,
            total_steps=env.current_step
        )
        
        print("\n=== Backtest Results ===")
        print(f"Symbol: {config['symbol']}")
        print(f"Period: {config['start_date']} to {config['end_date']}")
        print(f"Strategy: HOLD")
        print(f"Initial Capital: ${env.initial_capital:.2f}")
        print(f"Final Portfolio Value: ${metrics['Final Portfolio Value']:.2f}")
        print(f"Total Return: {metrics['Total Return (%)']:.2f}%")
        print(f"Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
        print(f"Sharpe Ratio: {metrics['Annualized Sharpe Ratio']:.3f}")
        print(f"Sortino Ratio: {metrics['Annualized Sortino Ratio']:.3f}")
        
        env.close()
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Error during backtest: {e}", exc_info=True)


if __name__ == "__main__":
    main() 