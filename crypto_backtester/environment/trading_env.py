"""
Trading environment for cryptocurrency trading using Gym-style API.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Union, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """Trading environment for cryptocurrency trading."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str,
        initial_balance: float = 10000,
        commission: float = 0.001,
        window_size: int = 30,
        reward_scaling: float = 1.0,
        max_steps: Optional[int] = None,
        reward_function: str = "sharpe",
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame containing price data with columns (open, high, low, close, volume)
            symbol: Trading symbol
            initial_balance: Initial balance in USD
            commission: Trading commission as a percentage
            window_size: Size of the observation window
            reward_scaling: Scaling factor for rewards
            max_steps: Maximum number of steps per episode (None = len(data))
            reward_function: Which reward function to use ('returns', 'sharpe', 'sortino', 'calmar')
        """
        super(TradingEnvironment, self).__init__()
        
        # Data
        self.data = data.reset_index(drop=True)
        self.symbol = symbol
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.reward_function = reward_function
        
        # Episode parameters
        self.current_step = 0
        self.max_steps = len(self.data) - self.window_size if max_steps is None else max_steps
        self.portfolio_value_history = []
        self.action_history = []
        self.returns_history = []
        
        # Define action and observation spaces
        # Actions: 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)
        
        # Feature dimension (price data + technical indicators + portfolio state)
        # Observations: normalized OHLCV data, technical indicators, portfolio state
        num_features = 5 + 15 + 3  # OHLCV + indicators + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, num_features), dtype=np.float32
        )
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            Initial observation
        """
        # Reset state
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_value_history = [self.portfolio_value]
        self.action_history = []
        self.returns_history = []
        self.buy_price = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment based on the action.
        
        Args:
            action: Action to take (0 = Sell, 1 = Hold, 2 = Buy)
            
        Returns:
            observation: Next observation
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Validate action
        if not self.action_space.contains(action):
            logger.warning(f"Invalid action: {action}. Using HOLD action instead.")
            action = 1  # Default to HOLD
        
        # Get current price
        current_price = self.data.iloc[self.current_step + self.window_size]["close"]
        
        # Take action
        self._take_action(action, current_price)
        
        # Update portfolio value
        self.portfolio_value = self.balance + self.shares_held * current_price
        self.portfolio_value_history.append(self.portfolio_value)
        
        # Calculate returns
        if len(self.portfolio_value_history) > 1:
            current_return = (self.portfolio_value_history[-1] / self.portfolio_value_history[-2]) - 1
        else:
            current_return = 0
        
        self.returns_history.append(current_return)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update current step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps - 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Prepare info
        info = {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "current_price": current_price,
            "current_step": self.current_step,
            "total_return": (self.portfolio_value / self.initial_balance) - 1,
        }
        
        return observation, reward, done, info
    
    def _take_action(self, action: int, current_price: float):
        """
        Execute the trading action.
        
        Args:
            action: Action to take (0 = Sell, 1 = Hold, 2 = Buy)
            current_price: Current asset price
        """
        # Record action
        self.action_history.append(action)
        
        # Sell
        if action == 0 and self.shares_held > 0:
            # Calculate sale amount and commission
            sale_amount = self.shares_held * current_price
            commission_fee = sale_amount * self.commission
            
            # Update balance and shares held
            self.balance += sale_amount - commission_fee
            self.shares_held = 0
            
            logger.debug(f"Step {self.current_step}: SELL {self.shares_held} shares at {current_price}")
        
        # Buy
        elif action == 2 and self.balance > 0:
            # Calculate maximum shares that can be bought
            max_shares = self.balance / (current_price * (1 + self.commission))
            
            # We'll use all available balance to buy shares
            self.shares_held += max_shares
            commission_fee = max_shares * current_price * self.commission
            self.balance = 0
            
            # Record buy price for reference
            self.buy_price = current_price
            
            logger.debug(f"Step {self.current_step}: BUY {max_shares} shares at {current_price}")
        
        # Hold - do nothing
        else:
            logger.debug(f"Step {self.current_step}: HOLD, current price: {current_price}")
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (state).
        
        Returns:
            Current observation as a numpy array
        """
        # Get data window
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size
        
        # Extract relevant data (OHLCV)
        data_window = self.data.iloc[start_idx:end_idx].copy()
        
        # Extract and normalize OHLCV data
        ohlcv_data = data_window[["open", "high", "low", "close", "volume"]].values
        
        # Normalize price data using first price in the window
        first_close = ohlcv_data[0, 3]  # First close price
        ohlcv_data[:, 0:4] = ohlcv_data[:, 0:4] / first_close - 1  # Normalize price data
        
        # Normalize volume data
        max_volume = ohlcv_data[:, 4].max()
        if max_volume > 0:
            ohlcv_data[:, 4] = ohlcv_data[:, 4] / max_volume
        
        # Get technical indicators (already in the data if extracted features)
        indicators = []
        indicator_columns = [
            col for col in data_window.columns
            if col not in ["open", "high", "low", "close", "volume", "timestamp", "date", "time"]
        ]
        
        if indicator_columns:
            indicators = data_window[indicator_columns].values
        else:
            # If no indicators in data, create empty array
            indicators = np.zeros((self.window_size, 15))
        
        # Create portfolio state features
        current_price = self.data.iloc[self.current_step + self.window_size - 1]["close"]
        portfolio_state = np.zeros((self.window_size, 3))
        
        # Portfolio state: [balance_ratio, shares_held_ratio, portfolio_value_ratio]
        balance_ratio = self.balance / self.initial_balance
        shares_value_ratio = (self.shares_held * current_price) / self.initial_balance
        portfolio_value_ratio = self.portfolio_value / self.initial_balance
        
        # Set the same portfolio state for all timesteps in the window
        portfolio_state[:, 0] = balance_ratio
        portfolio_state[:, 1] = shares_value_ratio
        portfolio_state[:, 2] = portfolio_value_ratio
        
        # Combine features
        observation = np.concatenate(
            [ohlcv_data, indicators, portfolio_state], axis=1
        ).astype(np.float32)
        
        return observation
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the selected reward function.
        
        Returns:
            Reward value
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        # Calculate different reward components
        if self.reward_function == "returns":
            # Simple returns based reward
            reward = self.returns_history[-1] * self.reward_scaling
        
        elif self.reward_function == "sharpe":
            # Sharpe ratio based reward (if we have enough data)
            if len(self.returns_history) >= 10:
                returns = np.array(self.returns_history[-10:])
                mean_return = np.mean(returns)
                std_return = np.std(returns) + 1e-6  # Avoid division by zero
                sharpe = mean_return / std_return
                reward = sharpe * self.reward_scaling
            else:
                reward = self.returns_history[-1] * self.reward_scaling
        
        elif self.reward_function == "sortino":
            # Sortino ratio based reward (if we have enough data)
            if len(self.returns_history) >= 10:
                returns = np.array(self.returns_history[-10:])
                mean_return = np.mean(returns)
                downside_returns = returns[returns < 0]
                downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
                sortino = mean_return / downside_deviation
                reward = sortino * self.reward_scaling
            else:
                reward = self.returns_history[-1] * self.reward_scaling
        
        elif self.reward_function == "calmar":
            # Calmar ratio based reward (if we have enough data)
            if len(self.portfolio_value_history) >= 10:
                returns = np.array(self.returns_history[-10:])
                mean_return = np.mean(returns)
                
                # Calculate max drawdown
                portfolio_values = np.array(self.portfolio_value_history[-10:])
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - peak) / peak
                max_drawdown = abs(np.min(drawdown)) + 1e-6  # Avoid division by zero
                
                calmar = mean_return / max_drawdown
                reward = calmar * self.reward_scaling
            else:
                reward = self.returns_history[-1] * self.reward_scaling
        
        else:
            # Default to returns
            reward = self.returns_history[-1] * self.reward_scaling
        
        return reward
    
    def render(self, mode="human"):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == "human":
            current_price = self.data.iloc[self.current_step + self.window_size - 1]["close"]
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price}")
            print(f"Balance: {self.balance}")
            print(f"Shares Held: {self.shares_held}")
            print(f"Portfolio Value: {self.portfolio_value}")
            print(f"Total Return: {(self.portfolio_value / self.initial_balance - 1) * 100:.2f}%")
            print("-" * 50)