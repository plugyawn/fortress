"""
Backtesting environment for cryptocurrency trading
"""

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional

from fortress.data.fetcher import fetch_binance_data
from fortress.indicators.technical import add_indicators
from fortress.metrics.performance import (
    calculate_sortino_ratio, 
    calculate_sharpe_ratio, 
    calculate_max_drawdown
)
from fortress.utils.helpers import (
    estimate_periods_per_year,
    validate_config,
    format_state_summary
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestingEnv(gym.Env):
    """
    Gymnasium environment for crypto backtesting.

    Action Space:
        Discrete(3): 0: Hold, 1: Buy (all-in), 2: Sell (close position)

    Observation Space:
        Box: Contains normalized lookback data (price/vol as % change, indicators scaled)
             and cyclical time features (day, hour). Portfolio state is in `info`.

    Reward:
        Differential Sortino Ratio (change in Sortino ratio from previous step).
    """
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 10}

    def __init__(self, config: Dict):
        super().__init__()

        # Process and validate config
        self.config = validate_config(config)
        self.symbol = self.config['symbol']
        self.interval = self.config['interval']
        self.initial_capital = self.config['initial_capital']
        self.commission = self.config['commission_perc']
        self.slippage = self.config['slippage_perc']
        self.lookback_window = self.config['lookback_window']
        self.sortino_period = self.config['reward_sortino_period']
        self.render_mode = self.config['render_mode']

        # Calculate risk-free rate per period
        self._periods_per_year = estimate_periods_per_year(self.interval)
        self.risk_free_rate_period = (1 + self.config['reward_risk_free_rate'])**(1/self._periods_per_year) - 1

        # --- 1. Load and Prepare Data ---
        self.df = self._load_and_prepare_data(self.config)
        if self.df.empty or len(self.df) <= self.lookback_window:
            raise ValueError("Dataframe is empty or too short after loading/processing.")

        self._validate_data()
        self.indicator_names = [col for col in self.df.columns if col not in \
                                ['open', 'high', 'low', 'close', 'volume', 'dayofweek', 'hour']]

        # --- 2. Define Spaces ---
        # Action: 0: Hold, 1: Buy, 2: Sell (simplified: buy max, sell all)
        self.action_space = spaces.Discrete(3)

        # Observation: Lookback window data (normalized) + time features
        # Price/Vol features: O, H, L, C, V (% change relative to start of window) -> 5 features
        # Indicator features: Scaled within the window -> len(self.indicator_names) features
        # Time features: dayofweek (scaled 0-1), hour (scaled 0-1) -> 2 features
        num_base_features = 5 + len(self.indicator_names) + 2
        obs_shape = (self.lookback_window, num_base_features)
        # Using flattened observation for compatibility with many standard RL agents
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_shape[0] * obs_shape[1],), 
            dtype=np.float32
        )

        # --- 3. Environment State Variables ---
        self.current_step = 0
        self.cash = 0.0
        self.position_size = 0.0  # Size in base asset (e.g., BTC)
        self.position_entry_price = 0.0
        self.position_entry_step = -1  # Step number when position was entered
        self.portfolio_value = 0.0
        self.portfolio_history = []  # List of (timestamp, value) tuples
        self.returns_history = pd.Series(dtype=np.float64)  # Track returns for Sortino
        self.trade_history = []
        self.last_sortino = 0.0  # For differential reward
        self.total_steps = len(self.df) - self.lookback_window - 1  # Allow one extra step for final price

        logger.info(f"Environment initialized for {self.symbol}. Data points: {len(self.df)}, Total steps: {self.total_steps}")
        logger.info(f"Observation shape (flattened): {self.observation_space.shape}")
        logger.info(f"Indicator features used: {self.indicator_names}")

    def _load_and_prepare_data(self, config: Dict) -> pd.DataFrame:
        """Loads data, adds indicators, and performs basic cleaning."""
        df = fetch_binance_data(config, config['force_data_refresh'])
        if df.empty: 
            return df
        df_with_indicators = add_indicators(df.copy(), config['indicators_config'])
        return df_with_indicators

    def _validate_data(self):
        """Check if necessary columns exist."""
        required = {'open', 'high', 'low', 'close', 'volume', 'dayofweek', 'hour'}
        if not required.issubset(self.df.columns):
            raise ValueError(f"Dataframe missing required columns. Needs: {required}, Has: {self.df.columns.tolist()}")

    def _get_current_data_index(self) -> int:
        """Gets the index in the DataFrame for the *end* of the current observation window."""
        # Observation window is [step, step + lookback - 1].
        # The action is decided based on this window.
        # The trade occurs at the open of the *next* candle (step + lookback).
        return self.current_step + self.lookback_window - 1  # Index of the last candle *in* the observation

    def _get_execution_price(self, action: int) -> Optional[float]:
        """
        Gets the price at which the trade would execute for the *next* timestep.
        Simulates execution at the next bar's open price + slippage.
        Returns None if data is unavailable (end of backtest).
        """
        execution_step_index = self._get_current_data_index() + 1  # Index of the candle where trade executes
        if execution_step_index >= len(self.df):
            return None  # No more data to execute

        execution_price = self.df['open'].iloc[execution_step_index]

        # Apply slippage
        if self.slippage > 0:
            if action == 1:  # Buying, price might slip up
                execution_price *= (1 + self.slippage)
            elif action == 2:  # Selling, price might slip down
                execution_price *= (1 - self.slippage)

        return execution_price

    def _calculate_portfolio_value(self, current_price: Optional[float] = None) -> float:
        """Calculates the current total value of the portfolio. Uses last close if price not provided."""
        if current_price is None:
            # Use the close price of the last candle in the current observation window for valuation
            last_obs_index = self._get_current_data_index()
            if last_obs_index < 0 or last_obs_index >= len(self.df): 
                return self.cash  # Edge case at start
            current_price = self.df['close'].iloc[last_obs_index]

        position_value = self.position_size * current_price
        return self.cash + position_value

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the observation array for the current step.
        Normalizes data within the lookback window.
        """
        start_idx = self.current_step
        end_idx = self.current_step + self.lookback_window  # Exclusive index

        obs_df = self.df.iloc[start_idx:end_idx].copy()

        # --- Normalize Features ---
        # 1. Price/Volume: Percentage change relative to the *first* value in the window
        price_vol_cols = ['open', 'high', 'low', 'close', 'volume']
        first_values = obs_df[price_vol_cols].iloc[0] + 1e-8  # Avoid division by zero
        for col in price_vol_cols:
            obs_df[col] = (obs_df[col] / first_values[col]) - 1.0

        # 2. Indicators: Scale using mean/stddev *within the window*
        indicator_cols = self.indicator_names
        if indicator_cols:  # Only scale if indicators exist
            means = obs_df[indicator_cols].mean(axis=0)
            stds = obs_df[indicator_cols].std(axis=0) + 1e-8  # Avoid division by zero
            obs_df[indicator_cols] = (obs_df[indicator_cols] - means) / stds

        # 3. Time Features: Scale to [0, 1]
        obs_df['dayofweek'] = obs_df['dayofweek'] / 6.0
        obs_df['hour'] = obs_df['hour'] / 23.0
        time_cols = ['dayofweek', 'hour']

        # --- Combine and Flatten ---
        feature_cols = price_vol_cols + indicator_cols + time_cols
        observation = obs_df[feature_cols].values.flatten().astype(np.float32)

        # Ensure shape matches definition
        if observation.shape[0] != self.observation_space.shape[0]:
            logger.error(f"Observation shape mismatch! Expected {self.observation_space.shape[0]}, got {observation.shape[0]}. Padding/Truncating.")
            expected_len = self.observation_space.shape[0]
            if len(observation) < expected_len:
                observation = np.pad(observation, (0, expected_len - len(observation)), 'constant', constant_values=0)
            elif len(observation) > expected_len:
                observation = observation[:expected_len]

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Returns supplementary information about the environment state."""
        current_data_idx = self._get_current_data_index()
        current_close_price = self.df['close'].iloc[current_data_idx] if current_data_idx >= 0 else np.nan
        current_portfolio_value = self._calculate_portfolio_value(current_close_price)

        unrealized_pnl = 0.0
        unrealized_pnl_perc = 0.0
        position_duration = 0
        if self.position_size > 1e-8:
            unrealized_pnl = (current_close_price - self.position_entry_price) * self.position_size
            unrealized_pnl_perc = (current_close_price / self.position_entry_price - 1) if self.position_entry_price > 1e-8 else 0.0
            position_duration = self.current_step - self.position_entry_step

        # Calculate rolling risk metrics
        rolling_sortino = 0.0
        rolling_sharpe = 0.0
        if len(self.returns_history) >= self.sortino_period:
            rolling_sortino = calculate_sortino_ratio(self.returns_history, self.risk_free_rate_period, self.sortino_period)
            rolling_sharpe = calculate_sharpe_ratio(self.returns_history, self.risk_free_rate_period, self.sortino_period)

        return {
            "symbol": self.symbol,
            "step": self.current_step,
            "timestamp": self.df.index[current_data_idx] if current_data_idx >= 0 else None,
            "cash": self.cash,
            "position_size": self.position_size,
            "position_entry_price": self.position_entry_price if self.position_size > 1e-8 else None,
            "position_duration_steps": position_duration if self.position_size > 1e-8 else 0,
            "market_price_close": current_close_price,  # Price at the end of the observation window
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_perc": unrealized_pnl_perc,
            "portfolio_value": current_portfolio_value,
            "rolling_sortino": rolling_sortino,  # Rolling metric based on past returns
            "rolling_sharpe": rolling_sharpe,    # Rolling metric based on past returns
            "last_trade": self.trade_history[-1] if self.trade_history else None,
            "sortino_period": self.sortino_period  # Add for context
        }

    def get_state_summary(self, info: Optional[Dict] = None) -> str:
        """Generates a concise text summary of the current state."""
        if info is None:
            info = self._get_info()
        return format_state_summary(info)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment."""
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_capital
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.position_entry_step = -1
        self.portfolio_value = self.initial_capital
        self.portfolio_history = [(self.df.index[self.lookback_window-1], self.initial_capital)]  # Start history at end of first obs window
        self.returns_history = pd.Series(dtype=np.float64)
        self.trade_history = []
        self.last_sortino = 0.0  # Reset initial Sortino for reward calc

        logger.info(f"Environment reset. Initial capital: ${self.initial_capital:.2f}")

        observation = self._get_observation()
        info = self._get_info()  # Get initial info state

        if self.render_mode == 'human':
            self._render_human(info)

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step."""
        terminated = False
        truncated = False  # Not used here, could be for episode length limit
        self.current_step += 1  # Increment step first

        # --- 1. Determine Execution Price ---
        execution_price = self._get_execution_price(action)
        if execution_price is None or self.current_step >= self.total_steps:
            # If no more data for execution or reached end, terminate
            terminated = True
            # Calculate final portfolio value based on last available close price
            last_obs_idx = self._get_current_data_index()
            last_close = self.df['close'].iloc[last_obs_idx] if last_obs_idx < len(self.df) else self.portfolio_history[-1][1] / (self.position_size if self.position_size > 1e-8 else 1)  # Estimate if needed
            self.portfolio_value = self._calculate_portfolio_value(last_close)
            self.portfolio_history.append((self.df.index[last_obs_idx] if last_obs_idx < len(self.df) else self.portfolio_history[-1][0] + pd.Timedelta(hours=1), self.portfolio_value))  # Approx timestamp if needed
            logger.info("End of data reached or execution price unavailable. Terminating.")
            reward = 0.0  # No reward on final step
            observation = self._get_observation()  # Get last valid observation
            info = self._get_info()
            info["status"] = "Terminated - End of Data"
            return observation, reward, terminated, truncated, info


        # --- 2. Portfolio Value BEFORE action (based on previous close) ---
        prev_close_idx = self._get_current_data_index()  # Index before the execution candle
        prev_close_price = self.df['close'].iloc[prev_close_idx]
        value_before_action = self._calculate_portfolio_value(prev_close_price)


        # --- 3. Execute Action ---
        action_taken_str = "Hold"
        trade_details = None

        # Action 1: Buy (if flat)
        if action == 1 and self.position_size < 1e-8:
            action_taken_str = "Buy"
            # Buy with all available cash, accounting for commission
            buy_amount_quote = self.cash * (1 - self.commission)
            self.position_size = buy_amount_quote / execution_price
            self.cash = 0.0
            self.position_entry_price = execution_price
            self.position_entry_step = self.current_step
            commission_paid = self.cash * self.commission  # Commission deducted before calculating size
            trade_details = {
                "type": "Buy", "step": self.current_step, "price": execution_price,
                "size": self.position_size, "commission": commission_paid
            }
            self.trade_history.append(trade_details)
            logger.debug(f"Step {self.current_step}: Executed Buy @ {execution_price:.2f}, Size: {self.position_size:.6f}")

        # Action 2: Sell (if long)
        elif action == 2 and self.position_size > 1e-8:
            action_taken_str = "Sell"
            # Sell entire position
            sell_amount_quote = self.position_size * execution_price
            commission_paid = sell_amount_quote * self.commission
            self.cash += sell_amount_quote - commission_paid
            profit = (execution_price - self.position_entry_price) * self.position_size - commission_paid
            size_sold = self.position_size  # Store before resetting
            entry_price = self.position_entry_price
            self.position_size = 0.0
            self.position_entry_price = 0.0
            self.position_entry_step = -1
            trade_details = {
                "type": "Sell", "step": self.current_step, "price": execution_price,
                "size": size_sold, "commission": commission_paid, "profit": profit,
                "entry_price": entry_price
            }
            self.trade_history.append(trade_details)
            logger.debug(f"Step {self.current_step}: Executed Sell @ {execution_price:.2f}, Profit: {profit:.2f}")

        # Action 0: Hold (or invalid action for current state)
        else:
            if action == 1: 
                action_taken_str = "Hold (Attempted Buy while Long)"
            elif action == 2: 
                action_taken_str = "Hold (Attempted Sell while Flat)"
            else: 
                action_taken_str = "Hold"
            # No change in position or cash


        # --- 4. Update Portfolio Value & History AFTER action ---
        # Use the close price of the candle *where execution happened* for valuation
        current_close_price = self.df['close'].iloc[prev_close_idx + 1]
        self.portfolio_value = self._calculate_portfolio_value(current_close_price)
        current_timestamp = self.df.index[prev_close_idx + 1]
        self.portfolio_history.append((current_timestamp, self.portfolio_value))

        # --- 5. Calculate Reward (Differential Sortino) ---
        # Calculate return for this step
        step_return = 0.0
        if value_before_action > 1e-8:  # Avoid division by zero if starting broke
            step_return = (self.portfolio_value / value_before_action) - 1.0
        self.returns_history = pd.concat([self.returns_history, pd.Series([step_return])], ignore_index=True)

        # Calculate current Sortino ratio
        current_sortino = 0.0
        if len(self.returns_history) >= self.sortino_period:
            current_sortino = calculate_sortino_ratio(self.returns_history, self.risk_free_rate_period, self.sortino_period)

        # Reward is the change in Sortino ratio
        reward = current_sortino - self.last_sortino
        self.last_sortino = current_sortino  # Update for next step


        # --- 6. Get Next Observation and Info ---
        observation = self._get_observation()
        info = self._get_info()
        info["action_taken"] = action_taken_str
        if trade_details: 
            info["trade_executed"] = trade_details
        info["reward"] = reward  # Include reward in info for clarity


        # --- 7. Render ---
        if self.render_mode == 'human':
            self._render_human(info)


        return observation, float(reward), terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            # Rendering is handled within step/reset for human mode currently
            pass

    def _render_human(self, info: Dict):
        """Simple text-based rendering using the info dict."""
        print(f"\n--- Step {info['step']}/{self.total_steps} ---")
        print(f"Timestamp: {info['timestamp']}")
        action_str = f"Action Taken: {info.get('action_taken', 'N/A')}"
        trade = info.get('trade_executed')
        if trade:
            profit_str = f", Profit: ${trade['profit']:.2f}" if 'profit' in trade else ""
            action_str += f" -> {trade['type']} @ ${trade['price']:.2f}{profit_str}"
        print(action_str)
        print(f"Market Close: ${info['market_price_close']:.2f}")
        print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
        print(f"Cash: ${info['cash']:.2f}")
        if info['position_size'] > 1e-8:
            print(f"Position: {info['position_size']:.6f} {self.symbol} @ Entry ${info['position_entry_price']:.2f} (Duration: {info['position_duration_steps']})")
            print(f"Unrealized PnL: ${info['unrealized_pnl']:.2f} ({info['unrealized_pnl_perc']*100:.2f}%)")
        else:
            print("Position: Flat")
        print(f"Reward (Diff Sortino): {info.get('reward', 0.0):.6f}")
        print(f"Rolling Sortino: {info['rolling_sortino']:.3f}")
        print("-" * 20)

    def close(self):
        """Close the environment and clean up resources."""
        logger.info("Closing Trading Environment.")
        self.portfolio_history = []  # Clear large data structures
        self.returns_history = pd.Series(dtype=np.float64)
        self.df = pd.DataFrame()  # Release dataframe memory 