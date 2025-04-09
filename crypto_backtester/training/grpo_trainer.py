"""
GRPO (Gradient-based Reward Policy Optimization) trainer for cryptocurrency trading agents.
This module implements the training loop and algorithm for fine-tuning trading agents.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from tqdm import tqdm
import wandb
import json
from dataclasses import dataclass

from crypto_backtester.environment.trading_env import TradingEnvironment
from crypto_backtester.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""
    # Environment parameters
    data_dir: str
    symbols: List[str]
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    window_size: int = 50
    initial_balance: float = 10000.0
    commission: float = 0.001
    reward_function: str = "sharpe"  # "returns", "sharpe", "sortino", "calmar"
    reward_scaling: float = 1.0
    max_steps: Optional[int] = None
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    kl_penalty: float = 0.1
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_minibatches: int = 4
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    gamma: float = 0.99
    clip_ratio: float = 0.2
    
    # Logging and saving
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 5
    model_dir: str = "models"
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "crypto-trading"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    
    # Model parameters
    model_type: str = "ppo"  # "ppo", "a2c", "sac", "td3"
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"  # "relu", "tanh", "leaky_relu"
    
    # Feature parameters
    use_technical_indicators: bool = True
    feature_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values and check for validity."""
        # Set default feature config if not provided
        if self.feature_config is None:
            self.feature_config = {
                "momentum_indicators": True,
                "volatility_indicators": True,
                "volume_indicators": True,
                "cycle_indicators": True,
                "pattern_recognition": False,
                "ma_types": ["SMA", "EMA"],
                "ma_periods": [7, 14, 21, 50],
                "rsi_periods": [7, 14, 21],
                "bbands_periods": [20],
            }
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate run name if not provided
        if self.run_name is None:
            self.run_name = f"grpo_{self.model_type}_{int(time.time())}"
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class GRPOTrainer:
    """
    Trainer class implementing Gradient-based Reward Policy Optimization.
    This is a policy gradient method for training RL agents.
    """
    
    def __init__(
        self, 
        agent: BaseAgent,
        config: TrainingConfig,
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            agent: The agent to train
            config: Training configuration
        """
        self.agent = agent
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move agent to device
        self.agent.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Set up environments for training and testing
        self._setup_environments()
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_environments(self) -> None:
        """Set up training and testing environments."""
        from crypto_backtester.features.technical_features import prepare_all_symbols_data
        
        # Create processed data directory
        processed_data_dir = os.path.join(self.config.data_dir, "processed")
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Prepare data for training period
        logger.info("Preparing training data...")
        train_data_files = prepare_all_symbols_data(
            self.config.data_dir,
            processed_data_dir,
            symbols=self.config.symbols,
            time_range={
                "start": self.config.train_start_date,
                "end": self.config.train_end_date
            },
            features_config=self.config.feature_config
        )
        
        # Prepare data for testing period
        logger.info("Preparing testing data...")
        test_data_files = prepare_all_symbols_data(
            self.config.data_dir,
            processed_data_dir,
            symbols=self.config.symbols,
            time_range={
                "start": self.config.test_start_date,
                "end": self.config.test_end_date
            },
            features_config=self.config.feature_config
        )
        
        # Create environments
        self.train_envs = []
        self.test_envs = []
        
        for symbol in self.config.symbols:
            if symbol in train_data_files:
                # Create training environment
                train_df = pd.read_csv(train_data_files[symbol])
                if 'timestamp' in train_df.columns:
                    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
                    train_df = train_df.set_index('timestamp')
                
                train_env = TradingEnvironment(
                    data=train_df,
                    symbol=symbol,
                    initial_balance=self.config.initial_balance,
                    commission=self.config.commission,
                    window_size=self.config.window_size,
                    reward_function=self.config.reward_function,
                    reward_scaling=self.config.reward_scaling,
                    max_steps=self.config.max_steps
                )
                self.train_envs.append(train_env)
            
            if symbol in test_data_files:
                # Create testing environment
                test_df = pd.read_csv(test_data_files[symbol])
                if 'timestamp' in test_df.columns:
                    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
                    test_df = test_df.set_index('timestamp')
                
                test_env = TradingEnvironment(
                    data=test_df,
                    symbol=symbol,
                    initial_balance=self.config.initial_balance,
                    commission=self.config.commission,
                    window_size=self.config.window_size,
                    reward_function=self.config.reward_function,
                    reward_scaling=self.config.reward_scaling,
                    max_steps=self.config.max_steps
                )
                self.test_envs.append(test_env)
        
        logger.info(f"Created {len(self.train_envs)} training environments and {len(self.test_envs)} testing environments")
    
    def _setup_logging(self) -> None:
        """Set up logging with Weights & Biases if specified."""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name,
                config=self.config.__dict__
            )
    
    def collect_rollouts(
        self, 
        envs: List[TradingEnvironment],
        num_steps: int
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]]]:
        """
        Collect rollouts from the environments.
        
        Args:
            envs: List of environments to collect rollouts from
            num_steps: Number of steps to collect for each environment
            
        Returns:
            Tuple of (batch_data, episode_infos)
                batch_data: Dictionary with keys 'obs', 'actions', 'values', 'rewards', 'dones', 'log_probs'
                episode_infos: List of episode information dictionaries
        """
        # Initialize storage
        batch_size = len(envs) * num_steps
        
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        episode_infos = []
        
        # Reset all environments
        obs_list = [env.reset() for env in envs]
        
        # Run for num_steps in each environment
        for step in range(num_steps):
            current_obs = torch.FloatTensor(np.stack(obs_list)).to(self.device)
            observations.append(current_obs)
            
            # Get action from agent
            with torch.no_grad():
                action_dist, value = self.agent(current_obs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            actions.append(action)
            values.append(value.squeeze(-1))
            log_probs.append(log_prob)
            
            # Execute actions in environments
            next_obs_list = []
            reward_list = []
            done_list = []
            info_list = []
            
            for i, (env, act) in enumerate(zip(envs, action.cpu().numpy())):
                next_obs, reward, done, info = env.step(act)
                next_obs_list.append(next_obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
                
                # Reset environment if done
                if done:
                    episode_infos.append(info)
                    next_obs = env.reset()
                    next_obs_list[i] = next_obs
            
            obs_list = next_obs_list
            rewards.append(torch.FloatTensor(reward_list).to(self.device))
            dones.append(torch.FloatTensor(done_list).to(self.device))
        
        # Get final value for bootstrapping
        with torch.no_grad():
            final_obs = torch.FloatTensor(np.stack(obs_list)).to(self.device)
            _, final_value = self.agent(final_obs)
            final_value = final_value.squeeze(-1)
        
        # Convert lists to tensors
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        
        # Compute advantages and returns using GAE
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        
        last_gae_lam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = final_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        returns = advantages + values
        
        # Reshape tensors to [batch_size, ...]
        batch_data = {
            "obs": observations.reshape(batch_size, *observations.shape[2:]),
            "actions": actions.reshape(batch_size, *actions.shape[2:]),
            "values": values.reshape(batch_size),
            "log_probs": log_probs.reshape(batch_size),
            "advantages": advantages.reshape(batch_size),
            "returns": returns.reshape(batch_size)
        }
        
        return batch_data, episode_infos
    
    def _update_policy(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the policy using the PPO algorithm.
        
        Args:
            batch_data: Dictionary with keys 'obs', 'actions', 'values', 'returns', 'advantages', 'log_probs'
            
        Returns:
            Dictionary with loss statistics
        """
        # Prepare data
        b_obs = batch_data["obs"]
        b_actions = batch_data["actions"]
        b_values = batch_data["values"]
        b_returns = batch_data["returns"]
        b_advantages = batch_data["advantages"]
        b_log_probs = batch_data["log_probs"]
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Create dataset and dataloader for mini-batch training
        dataset = TensorDataset(b_obs, b_actions, b_values, b_returns, b_advantages, b_log_probs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size // self.config.num_minibatches,
            shuffle=True
        )
        
        # Track statistics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        clip_fractions = []
        
        # Perform multiple epochs of updates
        for _ in range(self.config.ppo_epochs):
            for mb_obs, mb_actions, mb_values, mb_returns, mb_advantages, mb_log_probs in dataloader:
                # Get new action distribution and values
                action_dist, values = self.agent(mb_obs)
                
                # Calculate new log probabilities and entropy
                new_log_probs = action_dist.log_prob(mb_actions)
                entropy = action_dist.entropy().mean()
                
                # Calculate policy loss using clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_log_probs)
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * mb_advantages
                
                # Calculate policy gradient loss
                pg_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Calculate value loss
                value_loss = nn.MSELoss()(values.squeeze(-1), mb_returns)
                
                # Calculate KL divergence
                kl_div = (mb_log_probs - new_log_probs).mean()
                
                # Calculate clip fraction for monitoring
                clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
                
                # Calculate total loss
                loss = pg_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy + self.config.kl_penalty * kl_div
                
                # Perform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                
                # Record statistics
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                kl_divs.append(kl_div.item())
                clip_fractions.append(clip_fraction.item())
        
        # Return average loss statistics
        return {
            "policy_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "kl_divergence": np.mean(kl_divs),
            "clip_fraction": np.mean(clip_fractions)
        }
    
    def train(self, total_steps: int) -> Dict[str, Any]:
        """
        Train the agent for a specified number of steps.
        
        Args:
            total_steps: Total number of environment steps to train for
            
        Returns:
            Dictionary with training statistics
        """
        # Calculate steps per epoch
        steps_per_env = total_steps // len(self.train_envs)
        epoch_steps = min(1000, steps_per_env // self.config.epochs)
        
        # Initialize logging variables
        start_time = time.time()
        episode_rewards = []
        train_stats = {
            "train_returns": [],
            "train_lengths": [],
            "test_returns": [],
            "test_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "kl_divergences": [],
            "clip_fractions": [],
            "learning_rates": []
        }
        
        # Main training loop
        logger.info(f"Starting training for {total_steps} total steps")
        logger.info(f"Running {self.config.epochs} epochs with {epoch_steps} steps per epoch")
        
        pbar = tqdm(total=total_steps, desc="Training Progress")
        steps_completed = 0
        
        for epoch in range(self.config.epochs):
            # Collect rollouts
            batch_data, episode_infos = self.collect_rollouts(self.train_envs, epoch_steps)
            steps_in_epoch = len(self.train_envs) * epoch_steps
            steps_completed += steps_in_epoch
            pbar.update(steps_in_epoch)
            
            # Process episode information
            for info in episode_infos:
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                    train_stats["train_returns"].append(info["episode"]["r"])
                    train_stats["train_lengths"].append(info["episode"]["l"])
            
            # Update policy
            update_stats = self._update_policy(batch_data)
            
            # Log statistics
            train_stats["policy_losses"].append(update_stats["policy_loss"])
            train_stats["value_losses"].append(update_stats["value_loss"])
            train_stats["entropies"].append(update_stats["entropy"])
            train_stats["kl_divergences"].append(update_stats["kl_divergence"])
            train_stats["clip_fractions"].append(update_stats["clip_fraction"])
            train_stats["learning_rates"].append(self.optimizer.param_groups[0]["lr"])
            
            # Evaluate agent
            if epoch % self.config.eval_interval == 0:
                test_stats = self.evaluate()
                train_stats["test_returns"].extend(test_stats["returns"])
                train_stats["test_lengths"].extend(test_stats["lengths"])
                
                # Log to W&B
                if self.config.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "steps": steps_completed,
                        "train/policy_loss": update_stats["policy_loss"],
                        "train/value_loss": update_stats["value_loss"],
                        "train/entropy": update_stats["entropy"],
                        "train/kl_divergence": update_stats["kl_divergence"],
                        "train/clip_fraction": update_stats["clip_fraction"],
                        "train/return": np.mean(episode_rewards[-10:]) if episode_rewards else 0.0,
                        "test/return": np.mean(test_stats["returns"]),
                        "test/length": np.mean(test_stats["lengths"]),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"]
                    })
            
            # Log to console
            if epoch % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = steps_completed / elapsed_time
                remaining_steps = total_steps - steps_completed
                eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"Steps: {steps_completed}/{total_steps} | "
                    f"FPS: {steps_per_sec:.2f} | "
                    f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))} | "
                    f"Train Return: {np.mean(episode_rewards[-10:]):.2f} | "
                    f"Policy Loss: {update_stats['policy_loss']:.4f}"
                )
            
            # Save model
            if epoch % self.config.save_interval == 0:
                self.save(os.path.join(self.config.model_dir, f"{self.config.run_name}_epoch{epoch}.pt"))
        
        # Save final model
        self.save(os.path.join(self.config.model_dir, f"{self.config.run_name}_final.pt"))
        pbar.close()
        
        # Final evaluation
        final_stats = self.evaluate()
        logger.info(f"Training complete. Final test return: {np.mean(final_stats['returns']):.2f}")
        
        # Return training statistics
        return train_stats
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, List[float]]:
        """
        Evaluate the agent on the test environments.
        
        Args:
            num_episodes: Number of episodes to evaluate for each environment
            
        Returns:
            Dictionary with evaluation statistics (returns, lengths)
        """
        returns = []
        lengths = []
        
        for env in self.test_envs:
            for _ in range(num_episodes):
                obs = env.reset()
                done = False
                ep_return = 0.0
                ep_length = 0
                
                while not done:
                    # Get action from agent
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        action_dist, _ = self.agent(obs_tensor)
                        action = action_dist.sample().cpu().numpy()[0]
                    
                    # Execute action in environment
                    obs, reward, done, info = env.step(action)
                    ep_return += reward
                    ep_length += 1
                
                returns.append(ep_return)
                lengths.append(ep_length)
        
        return {
            "returns": returns,
            "lengths": lengths
        }
    
    def save(self, path: str) -> None:
        """
        Save agent model and training state.
        
        Args:
            path: Path to save the model to
        """
        state_dict = {
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.__dict__
        }
        torch.save(state_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load agent model and training state.
        
        Args:
            path: Path to load the model from
        """
        state_dict = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(state_dict["agent"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        logger.info(f"Model loaded from {path}")


def create_default_training_pipeline(
    data_dir: str,
    symbols: List[str],
    train_start_date: str,
    train_end_date: str,
    test_start_date: str,
    test_end_date: str,
    model_type: str = "ppo",
    agent_cls: Optional[type] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[GRPOTrainer, BaseAgent]:
    """
    Create a default training pipeline with sensible defaults.
    
    Args:
        data_dir: Directory containing CSV files with price data
        symbols: List of symbols to train on
        train_start_date: Start date for training data
        train_end_date: End date for training data
        test_start_date: Start date for testing data
        test_end_date: End date for testing data
        model_type: Type of model to use ("ppo", "a2c", "sac", "td3")
        agent_cls: Agent class to use (if None, will be determined from model_type)
        config_overrides: Configuration overrides
        
    Returns:
        Tuple of (trainer, agent)
    """
    from crypto_backtester.agents.ppo_agent import PPOAgent
    
    # Create default configuration
    config = TrainingConfig(
        data_dir=data_dir,
        symbols=symbols,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        model_type=model_type
    )
    
    # Apply configuration overrides
    if config_overrides is not None:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Determine observation space size
    # This is a rough estimate - in practice, this would be determined from the actual environment
    sample_df = pd.read_csv(os.path.join(data_dir, f"{symbols[0]}.csv"))
    from crypto_backtester.features.technical_features import add_technical_features
    sample_df = add_technical_features(sample_df, config.feature_config)
    
    num_features = len(sample_df.columns)
    
    # Create agent
    if agent_cls is None:
        if model_type.lower() == "ppo":
            agent_cls = PPOAgent
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    agent = agent_cls(
        input_dim=num_features * config.window_size + 3,  # +3 for portfolio features
        hidden_dim=config.hidden_size,
        num_actions=3,  # buy, hold, sell
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    # Create trainer
    trainer = GRPOTrainer(agent, config)
    
    return trainer, agent