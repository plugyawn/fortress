# -*- coding: utf-8 -*-
"""
Enhanced Blazing Fast One-File Crypto Backtesting Environment for LLM Agents

Features:
- Binance data fetching with caching
- Gymnasium env interface (reset, step)
- Realistic execution (next open + slippage)
- Rich context in `info` dict for LLM
- Enhanced observation space (time features, % change norm)
- Advanced reward shaping (differential Sortino ratio)
- Detailed performance metrics (Sharpe, Sortino, Drawdown)
- State summarization helper for LLM prompts
- Configurable indicators via pandas_ta
"""

import os
import time
import requests
import hashlib
import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from binance.client import Client
import pandas_ta as ta # For indicators

# --- Configuration & Constants ---

# It's STRONGLY recommended to use environment variables or a secure config method
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "YOUR_API_KEY_READONLY") # Read-only is sufficient for backtesting data
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "YOUR_API_SECRET_READONLY")

# Backtesting Parameters
DEFAULT_CONFIG = {
    "symbol": "BTCUSDT",
    "interval": Client.KLINE_INTERVAL_1HOUR,
    "start_date": "2022-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 10000.0,
    "commission_perc": 0.001, # 0.1%
    "slippage_perc": 0.0005, # 0.05% slippage simulation (applied to execution price)
    "lookback_window": 50, # Number of past steps in observation
    "indicators_config": [
        {"kind": "rsi", "length": 14},
        {"kind": "macd"},
        {"kind": "bbands", "length": 20, "std": 2},
        {"kind": "atr", "length": 14},
        {"kind": "ema", "length": 50},
        {"kind": "ema", "length": 200},
        {"kind": "adx", "length": 14},
        # Add more indicators here
    ],
    "reward_risk_free_rate": 0.0, # Annualized risk-free rate for Sharpe/Sortino (e.g., 0.02 for 2%)
    "reward_sortino_period": 120, # Lookback period for Sortino calculation in reward
    "force_data_refresh": False,
    "cache_dir": "crypto_data_cache_v2", # Separate cache dir
    "render_mode": 'human', # 'human' or 'none'
}

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Binance Data Fetching (Cached) ---

def generate_cache_filename(symbol: str, interval: str, start_date: str, end_date: str, cache_dir: str) -> str:
    """Generates a unique filename for caching based on request parameters."""
    params_str = f"{symbol}_{interval}_{start_date}_{end_date}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{params_hash}.pkl")

def fetch_binance_data(config: Dict, force_refresh: bool = False) -> pd.DataFrame:
    """Fetches historical OHLCV data from Binance, using caching."""
    symbol = config['symbol']
    interval = config['interval']
    start_date = config['start_date']
    end_date = config['end_date']
    cache_dir = config['cache_dir']
    api_key = BINANCE_API_KEY
    api_secret = BINANCE_API_SECRET

    cache_filename = generate_cache_filename(symbol, interval, start_date, end_date, cache_dir)

    if not force_refresh and os.path.exists(cache_filename):
        try:
            logger.info(f"Loading cached data from {cache_filename}")
            df = pd.read_pickle(cache_filename)
            return df
        except Exception as e:
            logger.warning(f"Could not load cache file {cache_filename}: {e}. Fetching fresh data.")

    logger.info(f"Fetching data for {symbol} ({interval}) from {start_date} to {end_date}")
    client = Client(api_key, api_secret)

    try:
        klines_generator = client.get_historical_klines_generator(symbol, interval, start_date, end_date)
        all_klines = list(klines_generator)

        if not all_klines:
            logger.warning(f"No data found for {symbol} in the specified range.")
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Number of Trades',
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ])

        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Open Time', inplace=True)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # Keep only OHLCV + Volume + Trades for simplicity unless others needed
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = [col.lower() for col in df.columns] # Use lowercase

        try:
            df.to_pickle(cache_filename)
            logger.info(f"Data cached to {cache_filename}")
        except Exception as e:
            logger.error(f"Failed to save cache file {cache_filename}: {e}")

        return df

    except Exception as e:
        logger.error(f"An error occurred during data fetching: {e}")
        return pd.DataFrame()

# --- Indicator Calculation ---

def add_indicators(df: pd.DataFrame, indicators_config: List[Dict]) -> pd.DataFrame:
    """Adds technical indicators using pandas_ta."""
    if df.empty: return df
    logger.info(f"Calculating {len(indicators_config)} indicator group(s)...")

    # Ensure OHLCV are present and lowercase
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    strategy = ta.Strategy(name="LLM_Crypto_Strategy", ta=indicators_config)
    df.ta.strategy(strategy)

    # Add time-based features (useful for LLMs)
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour

    initial_len = len(df)
    df.dropna(inplace=True) # Drop rows with NaNs from indicator calculation
    logger.info(f"Dropped {initial_len - len(df)} rows with NaNs after indicator calculation.")
    return df

# --- Risk & Performance Calculation Helpers ---

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate_period: float, period: int) -> float:
    """Calculates Sortino Ratio for a given period."""
    if len(returns) < period: return 0.0

    relevant_returns = returns.iloc[-period:]
    mean_return = relevant_returns.mean()
    # Calculate downside deviation
    downside_returns = relevant_returns[relevant_returns < risk_free_rate_period]
    if len(downside_returns) <= 1: return 0.0 # Need >1 point to calculate std dev
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns - risk_free_rate_period)))
    if downside_deviation < 1e-8: return 0.0 # Avoid division by zero

    sortino = (mean_return - risk_free_rate_period) / downside_deviation
    return sortino if not np.isnan(sortino) else 0.0

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate_period: float, period: int) -> float:
    """Calculates Sharpe Ratio for a given period."""
    if len(returns) < period: return 0.0

    relevant_returns = returns.iloc[-period:]
    mean_return = relevant_returns.mean()
    std_dev = relevant_returns.std()
    if std_dev < 1e-8 or len(relevant_returns) <= 1 : return 0.0

    sharpe = (mean_return - risk_free_rate_period) / std_dev
    return sharpe if not np.isnan(sharpe) else 0.0

def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculates the maximum drawdown from a series of portfolio values."""
    if len(portfolio_values) < 2: return 0.0
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() # Minimum value represents the largest drawdown percentage
    return abs(max_drawdown) if not np.isnan(max_drawdown) else 0.0


# --- Crypto Backtesting Environment for LLM ---

class CryptoTradingEnvLLM(gym.Env):
    """
    Gymnasium environment for crypto backtesting, enhanced for LLM agent interaction.

    Action Space:
        Discrete(3): 0: Hold, 1: Buy (all-in), 2: Sell (close position)

    Observation Space:
        Box: Contains normalized lookback data (price/vol as % change, indicators scaled)
             and cyclical time features (day, hour). Portfolio state is in `info`.

    Reward:
        Differential Sortino Ratio (change in Sortino ratio from previous step).
    """
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 10}

    def __init__(self, config: Dict = DEFAULT_CONFIG):
        super().__init__()

        self.config = config
        self.symbol = config['symbol']
        self.interval = config['interval']
        self.initial_capital = config['initial_capital']
        self.commission = config['commission_perc']
        self.slippage = config['slippage_perc']
        self.lookback_window = config['lookback_window']
        self.sortino_period = config['reward_sortino_period']
        self.render_mode = config['render_mode']

        # Calculate risk-free rate per period
        self._periods_per_year = self._estimate_periods_per_year(config['interval'])
        self.risk_free_rate_period = (1 + config['reward_risk_free_rate'])**(1/self._periods_per_year) - 1

        # --- 1. Load and Prepare Data ---
        self.df = self._load_and_prepare_data(config)
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape[0] * obs_shape[1],), dtype=np.float32)
        # If using CNN/Transformer agent, keep shape=obs_shape

        # --- 3. Environment State Variables ---
        self.current_step = 0
        self.cash = 0.0
        self.position_size = 0.0 # Size in base asset (e.g., BTC)
        self.position_entry_price = 0.0
        self.position_entry_step = -1 # Step number when position was entered
        self.portfolio_value = 0.0
        self.portfolio_history = [] # List of (timestamp, value) tuples
        self.returns_history = pd.Series(dtype=np.float64) # Track returns for Sortino
        self.trade_history = []
        self.last_sortino = 0.0 # For differential reward
        self.total_steps = len(self.df) - self.lookback_window - 1 # Allow one extra step for final price

        logger.info(f"Environment initialized for {self.symbol}. Data points: {len(self.df)}, Total steps: {self.total_steps}")
        logger.info(f"Observation shape (flattened): {self.observation_space.shape}")
        logger.info(f"Indicator features used: {self.indicator_names}")

    def _estimate_periods_per_year(self, interval: str) -> float:
        """Estimates the number of intervals in a year."""
        interval_map = {
            '1m': 60*24*365.25, '3m': 20*24*365.25, '5m': 12*24*365.25, '15m': 4*24*365.25,
            '30m': 2*24*365.25, '1h': 24*365.25, '2h': 12*365.25, '4h': 6*365.25,
            '6h': 4*365.25, '8h': 3*365.25, '12h': 2*365.25, '1d': 365.25, '1w': 52
        }
        if interval not in interval_map:
            logger.warning(f"Interval '{interval}' not recognized for period estimation, defaulting to 365.25 (daily).")
            return 365.25
        return interval_map[interval]

    def _load_and_prepare_data(self, config: Dict) -> pd.DataFrame:
        """Loads data, adds indicators, and performs basic cleaning."""
        df = fetch_binance_data(config, config['force_data_refresh'])
        if df.empty: return df
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
        return self.current_step + self.lookback_window -1 # Index of the last candle *in* the observation

    def _get_execution_price(self, action: int) -> Optional[float]:
        """
        Gets the price at which the trade would execute for the *next* timestep.
        Simulates execution at the next bar's open price + slippage.
        Returns None if data is unavailable (end of backtest).
        """
        execution_step_index = self._get_current_data_index() + 1 # Index of the candle where trade executes
        if execution_step_index >= len(self.df):
            return None # No more data to execute

        execution_price = self.df['open'].iloc[execution_step_index]

        # Apply slippage
        if self.slippage > 0:
            if action == 1: # Buying, price might slip up
                execution_price *= (1 + self.slippage)
            elif action == 2: # Selling, price might slip down
                execution_price *= (1 - self.slippage)

        return execution_price

    def _calculate_portfolio_value(self, current_price: Optional[float] = None) -> float:
        """Calculates the current total value of the portfolio. Uses last close if price not provided."""
        if current_price is None:
            # Use the close price of the last candle in the current observation window for valuation
             last_obs_index = self._get_current_data_index()
             if last_obs_index < 0 or last_obs_index >= len(self.df): return self.cash # Edge case at start
             current_price = self.df['close'].iloc[last_obs_index]

        position_value = self.position_size * current_price
        return self.cash + position_value

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the observation array for the current step.
        Normalizes data within the lookback window.
        """
        start_idx = self.current_step
        end_idx = self.current_step + self.lookback_window # Exclusive index

        obs_df = self.df.iloc[start_idx:end_idx].copy()

        # --- Normalize Features ---
        # 1. Price/Volume: Percentage change relative to the *first* value in the window
        price_vol_cols = ['open', 'high', 'low', 'close', 'volume']
        first_values = obs_df[price_vol_cols].iloc[0] + 1e-8 # Avoid division by zero
        for col in price_vol_cols:
            obs_df[col] = (obs_df[col] / first_values[col]) - 1.0

        # 2. Indicators: Scale using mean/stddev *within the window*
        indicator_cols = self.indicator_names
        if indicator_cols: # Only scale if indicators exist
            means = obs_df[indicator_cols].mean(axis=0)
            stds = obs_df[indicator_cols].std(axis=0) + 1e-8 # Avoid division by zero
            obs_df[indicator_cols] = (obs_df[indicator_cols] - means) / stds

        # 3. Time Features: Scale to [0, 1] or [-1, 1] (using sin/cos is better for cyclical)
        # Simple scaling [0, 1] for now:
        obs_df['dayofweek'] = obs_df['dayofweek'] / 6.0
        obs_df['hour'] = obs_df['hour'] / 23.0
        time_cols = ['dayofweek', 'hour']

        # --- Combine and Flatten ---
        feature_cols = price_vol_cols + indicator_cols + time_cols
        observation = obs_df[feature_cols].values.flatten().astype(np.float32)

        # Ensure shape matches definition (handle potential edge cases if needed, though unlikely with proper indexing)
        if observation.shape[0] != self.observation_space.shape[0]:
             logger.error(f"Observation shape mismatch! Expected {self.observation_space.shape[0]}, got {observation.shape[0]}. Padding/Truncating.")
             expected_len = self.observation_space.shape[0]
             if len(observation) < expected_len:
                 observation = np.pad(observation, (0, expected_len - len(observation)), 'constant', constant_values=0) # Pad with 0
             elif len(observation) > expected_len:
                 observation = observation[:expected_len] # Truncate

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Returns supplementary information (rich context for LLM)."""
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

        # Calculate rolling risk metrics (optional, can be expensive)
        rolling_sortino = 0.0
        rolling_sharpe = 0.0
        if len(self.returns_history) >= self.sortino_period:
             rolling_sortino = calculate_sortino_ratio(self.returns_history, self.risk_free_rate_period, self.sortino_period)
             rolling_sharpe = calculate_sharpe_ratio(self.returns_history, self.risk_free_rate_period, self.sortino_period) # Use same period for consistency

        return {
            "step": self.current_step,
            "timestamp": self.df.index[current_data_idx] if current_data_idx >=0 else None,
            "cash": self.cash,
            "position_size": self.position_size,
            "position_entry_price": self.position_entry_price if self.position_size > 1e-8 else None,
            "position_duration_steps": position_duration if self.position_size > 1e-8 else 0,
            "market_price_close": current_close_price, # Price at the end of the observation window
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_perc": unrealized_pnl_perc,
            "portfolio_value": current_portfolio_value,
            "rolling_sortino": rolling_sortino, # Rolling metric based on past returns
            "rolling_sharpe": rolling_sharpe,   # Rolling metric based on past returns
            "last_trade": self.trade_history[-1] if self.trade_history else None
        }

    def get_llm_prompt_context(self, info: Optional[Dict] = None) -> str:
        """Generates a concise text summary of the current state for an LLM prompt."""
        if info is None:
            info = self._get_info()

        summary = f"--- Trading Environment State (Step {info['step']}) ---\n"
        summary += f"Timestamp: {info['timestamp']}\n"
        summary += f"Portfolio Value: ${info['portfolio_value']:.2f}\n"
        summary += f"Cash: ${info['cash']:.2f}\n"

        if info['position_size'] > 1e-8:
            summary += f"Position: LONG {self.symbol}\n"
            summary += f"  Size: {info['position_size']:.6f}\n"
            summary += f"  Entry Price: ${info['position_entry_price']:.2f}\n"
            summary += f"  Duration: {info['position_duration_steps']} steps\n"
            summary += f"  Current Market Price: ${info['market_price_close']:.2f}\n"
            summary += f"  Unrealized PnL: ${info['unrealized_pnl']:.2f} ({info['unrealized_pnl_perc']*100:.2f}%)\n"
        else:
            summary += "Position: FLAT (no open position)\n"

        summary += f"Risk Metrics (Rolling {self.sortino_period}-step):\n"
        summary += f"  Sortino Ratio: {info['rolling_sortino']:.3f}\n"
        # summary += f"  Sharpe Ratio: {info['rolling_sharpe']:.3f}\n" # Can add if needed

        last_trade = info.get('last_trade')
        if last_trade:
             trade_type = last_trade['type']
             trade_price = last_trade['price']
             trade_step = last_trade['step']
             profit_str = ""
             if 'profit' in last_trade:
                 profit_str = f", Profit: ${last_trade['profit']:.2f}"
             summary += f"Last Trade: {trade_type} at ${trade_price:.2f} (Step {trade_step}{profit_str})\n"

        summary += "--- End State ---"
        # NOTE: The numerical observation array (from _get_observation) should ideally be passed
        # separately to the LLM if it can process numerical sequences, or features extracted from it.
        # This text prompt provides the high-level context.
        return summary


    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment."""
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_capital
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.position_entry_step = -1
        self.portfolio_value = self.initial_capital
        self.portfolio_history = [(self.df.index[self.lookback_window-1], self.initial_capital)] # Start history at end of first obs window
        self.returns_history = pd.Series(dtype=np.float64)
        self.trade_history = []
        self.last_sortino = 0.0 # Reset initial Sortino for reward calc

        logger.info(f"Environment reset. Initial capital: ${self.initial_capital:.2f}")

        observation = self._get_observation()
        info = self._get_info() # Get initial info state

        if self.render_mode == 'human':
            self._render_human(info)

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step."""
        terminated = False
        truncated = False # Not used here, could be for episode length limit
        self.current_step += 1 # Increment step first

        # --- 1. Determine Execution Price ---
        execution_price = self._get_execution_price(action)
        if execution_price is None or self.current_step >= self.total_steps:
            # If no more data for execution or reached end, terminate
            terminated = True
            # Calculate final portfolio value based on last available close price
            last_obs_idx = self._get_current_data_index()
            last_close = self.df['close'].iloc[last_obs_idx] if last_obs_idx < len(self.df) else self.portfolio_history[-1][1] / (self.position_size if self.position_size > 1e-8 else 1) # Estimate if needed
            self.portfolio_value = self._calculate_portfolio_value(last_close)
            self.portfolio_history.append((self.df.index[last_obs_idx] if last_obs_idx < len(self.df) else self.portfolio_history[-1][0] + pd.Timedelta(hours=1), self.portfolio_value)) # Approx timestamp if needed
            logger.info("End of data reached or execution price unavailable. Terminating.")
            reward = 0.0 # No reward on final step
            observation = self._get_observation() # Get last valid observation
            info = self._get_info()
            info["status"] = "Terminated - End of Data"
            return observation, reward, terminated, truncated, info


        # --- 2. Portfolio Value BEFORE action (based on previous close) ---
        prev_close_idx = self._get_current_data_index() # Index before the execution candle
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
            commission_paid = self.cash * self.commission # Commission deducted before calculating size
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
            size_sold = self.position_size # Store before resetting
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
            if action == 1: action_taken_str = "Hold (Attempted Buy while Long)"
            elif action == 2: action_taken_str = "Hold (Attempted Sell while Flat)"
            else: action_taken_str = "Hold"
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
        if value_before_action > 1e-8: # Avoid division by zero if starting broke
            step_return = (self.portfolio_value / value_before_action) - 1.0
        self.returns_history = pd.concat([self.returns_history, pd.Series([step_return])], ignore_index=True)

        # Calculate current Sortino ratio
        current_sortino = 0.0
        if len(self.returns_history) >= self.sortino_period:
             current_sortino = calculate_sortino_ratio(self.returns_history, self.risk_free_rate_period, self.sortino_period)
        else:
            # Optional: Provide small reward/penalty based on simple return if not enough history?
            # Or just give 0 reward until enough history builds up. Let's use 0.
            pass


        # Reward is the change in Sortino ratio
        reward = current_sortino - self.last_sortino
        self.last_sortino = current_sortino # Update for next step

        # Optional: Add small penalty for commissions to discourage over-trading?
        # if trade_details: reward -= (trade_details['commission'] / self.portfolio_value) * 0.1 # Example small penalty


        # --- 6. Get Next Observation and Info ---
        observation = self._get_observation()
        info = self._get_info()
        info["action_taken"] = action_taken_str
        if trade_details: info["trade_executed"] = trade_details
        info["reward"] = reward # Include reward in info for clarity


        # --- 7. Render ---
        if self.render_mode == 'human':
            self._render_human(info)


        return observation, float(reward), terminated, truncated, info

    def render(self):
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
        logger.info("Closing Crypto Trading Environment.")
        self.portfolio_history = [] # Clear large data structures
        self.returns_history = pd.Series(dtype=np.float64)
        self.df = pd.DataFrame() # Release dataframe memory


# --- Performance Metrics Calculation ---

def calculate_performance_metrics(
    portfolio_history: List[Tuple[pd.Timestamp, float]],
    trade_history: List[Dict],
    initial_capital: float,
    risk_free_rate_annual: float,
    periods_per_year: float,
    total_steps: int
) -> Dict:
    """Calculates comprehensive performance metrics."""
    if not portfolio_history: return {"Error": "No portfolio history available."}

    portfolio_df = pd.DataFrame(portfolio_history, columns=['Timestamp', 'Value']).set_index('Timestamp')
    if len(portfolio_df) < 2 : return {"Error": "Portfolio history too short."}

    final_value = portfolio_df['Value'].iloc[-1]
    total_return_perc = (final_value / initial_capital - 1) * 100

    # Calculate returns for risk metrics
    returns = portfolio_df['Value'].pct_change().dropna()
    if len(returns) < 2: return {"Warning": "Not enough returns data for risk metrics."}

    risk_free_rate_period = (1 + risk_free_rate_annual)**(1/periods_per_year) - 1

    # Sharpe Ratio (using all returns)
    mean_return_period = returns.mean()
    std_dev_period = returns.std()
    sharpe_ratio = 0.0
    if std_dev_period > 1e-8:
         sharpe_ratio = (mean_return_period - risk_free_rate_period) / std_dev_period * np.sqrt(periods_per_year) # Annualize
         sharpe_ratio = sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0


    # Sortino Ratio (using all returns)
    downside_returns = returns[returns < risk_free_rate_period]
    sortino_ratio = 0.0
    if len(downside_returns) > 1:
        downside_deviation = np.sqrt(np.mean(np.square(downside_returns - risk_free_rate_period)))
        if downside_deviation > 1e-8:
            sortino_ratio = (mean_return_period - risk_free_rate_period) / downside_deviation * np.sqrt(periods_per_year) # Annualize
            sortino_ratio = sortino_ratio if not np.isnan(sortino_ratio) else 0.0


    # Max Drawdown
    max_drawdown_perc = calculate_max_drawdown(portfolio_df['Value']) * 100

    # Trade Stats
    closing_trades = [t for t in trade_history if t['type'] == 'Sell']
    num_trades = len(closing_trades)
    wins = [t for t in closing_trades if t.get('profit', 0) > 0]
    losses = [t for t in closing_trades if t.get('profit', 0) <= 0]
    win_rate = len(wins) / num_trades * 100 if num_trades > 0 else 0
    total_profit = sum(t['profit'] for t in wins)
    total_loss = abs(sum(t['profit'] for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    avg_profit_per_win = total_profit / len(wins) if wins else 0
    avg_loss_per_loss = total_loss / len(losses) if losses else 0

    return {
        "Initial Capital": initial_capital,
        "Final Portfolio Value": final_value,
        "Total Return (%)": total_return_perc,
        "Max Drawdown (%)": max_drawdown_perc,
        "Annualized Sharpe Ratio": sharpe_ratio,
        "Annualized Sortino Ratio": sortino_ratio,
        "Total Steps": total_steps,
        "Number of Trades": num_trades,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Average Winning Trade ($)": avg_profit_per_win,
        "Average Losing Trade ($)": avg_loss_per_loss,
        "Total Profit ($)": total_profit,
        "Total Loss ($)": total_loss,
        "Total Commissions ($)": sum(t['commission'] for t in trade_history),
    }


# --- Example Usage with LLM Interaction Simulation ---

if __name__ == "__main__":
    print("--- Starting Enhanced Crypto Backtesting Environment Example ---")

    # --- Environment Setup ---
    # Use a slightly shorter period for faster demo run
    env_config = DEFAULT_CONFIG.copy()
    env_config.update({
        "symbol": "ETHUSDT",
        "interval": Client.KLINE_INTERVAL_4HOUR, # Try different timeframe
        "start_date": "2023-01-01",
        "end_date": "2023-09-30",
        "lookback_window": 30,
        "sortino_period": 60, # Adjust based on timeframe/lookback
        "render_mode": 'human' # 'human' for detailed prints, 'none' for silent
    })

    env = CryptoTradingEnvLLM(config=env_config)

    # --- LLM Agent Simulation Loop ---
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward_accumulated = 0.0
    step_count = 0

    while not terminated and not truncated:
        step_count += 1
        # 1. Get Context for LLM
        llm_context = env.get_llm_prompt_context(info)

        # ** THIS IS WHERE THE LLM INTERACTION HAPPENS **
        # In a real scenario: send `llm_context` and maybe parts of `observation`
        # (or a summary/interpretation) to the LLM API.
        print(f"\n--- Step {info['step']} LLM Input ---")
        print(llm_context)
        # print(f"Observation sample (first 10): {observation[:10]}") # LLM might need numerical data too

        # Placeholder for LLM decision logic:
        # Replace this with your actual LLM call and response parsing.
        # Example: Simple logic based on unrealized PnL % (NOT a good strategy!)
        action = 0 # Default Hold
        if info['position_size'] < 1e-8: # If flat
            # Dummy logic: Buy if rolling Sortino is positive? (Needs more sophistication)
            if info['rolling_sortino'] > 0.1:
                 action = 1 # Buy
                 llm_reasoning = "LLM decides to BUY (Rolling Sortino positive)"
            else:
                 llm_reasoning = "LLM decides to HOLD (Flat, Sortino not compelling)"
        else: # If long
            # Dummy logic: Sell if unrealized loss > 2% or profit > 5%
            if info['unrealized_pnl_perc'] < -0.02:
                action = 2 # Sell (Stop Loss)
                llm_reasoning = "LLM decides to SELL (Stop Loss Triggered)"
            elif info['unrealized_pnl_perc'] > 0.05:
                action = 2 # Sell (Take Profit)
                llm_reasoning = "LLM decides to SELL (Take Profit Triggered)"
            else:
                llm_reasoning = "LLM decides to HOLD (Position Open)"

        # action = env.action_space.sample() # Alternative: Random Agent
        # llm_reasoning = "LLM decides RANDOMLY"

        print(f">>> LLM Reasoning: {llm_reasoning}")
        print(f">>> LLM Action Choice: {action} ({ {0: 'Hold', 1: 'Buy', 2: 'Sell'}[action] })")

        # 2. Execute action in environment
        try:
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward_accumulated += reward
        except Exception as e:
            logger.error(f"Error during env.step(): {e}", exc_info=True)
            terminated = True # Stop simulation on error

        # Optional delay for human reading
        if env.render_mode == 'human':
             time.sleep(0.1) # Small delay

        # Safety break for debugging runaway loops
        # if step_count > 5000:
        #    print("Stopping early due to step limit.")
        #    terminated = True


    # --- Final Performance ---
    print("\n--- Backtest Finished ---")

    # Use the collected history for metrics
    performance_metrics = calculate_performance_metrics(
        portfolio_history=env.portfolio_history,
        trade_history=env.trade_history,
        initial_capital=env.initial_capital,
        risk_free_rate_annual=env.config['reward_risk_free_rate'],
        periods_per_year=env._periods_per_year,
        total_steps=env.current_step # Actual steps taken
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