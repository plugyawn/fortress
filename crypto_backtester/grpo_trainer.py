"""
GRPO Trainer for LLM-based trading agents.

This module implements Generalized Regression for Policy Optimization (GRPO)
for fine-tuning LLM trading agents.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

# Import local modules
from crypto_backtester.utils.data_preparation import prepare_all_symbols_data, extract_features
from crypto_backtester.environment.trading_env import TradingEnvironment
from crypto_backtester.agents.llm_agent import LLMAgent
from crypto_backtester.metrics.performance import calculate_metrics

logger = logging.getLogger(__name__)

class GRPOTrainer:
    """Trainer class implementing GRPO for LLM trading agents."""
    
    def __init__(
        self,
        base_model_name: str,
        data_dir: str,
        output_dir: str,
        train_ratio: float = 0.75,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        alpha: float = 0.2,  # GRPO alpha parameter
        gamma: float = 0.99,  # Discount factor
        gae_lambda: float = 0.95,  # GAE parameter
        clip_ratio: float = 0.2,  # PPO clip parameter
        max_grad_norm: float = 1.0,
        device: str = None,
        seed: int = 42,
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            base_model_name: Name of the base LLM model
            data_dir: Directory containing CSV data files
            output_dir: Directory to save model checkpoints and logs
            train_ratio: Ratio of data to use for training vs validation
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            alpha: GRPO alpha parameter for balancing exploration/exploitation
            gamma: Discount factor for rewards
            gae_lambda: Generalized Advantage Estimation lambda parameter
            clip_ratio: PPO clip ratio for policy updates
            max_grad_norm: Maximum gradient norm for gradient clipping
            device: Device to use for training (cuda/cpu)
            seed: Random seed for reproducibility
        """
        self.base_model_name = base_model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.max_grad_norm = max_grad_norm
        
        # Set random seed
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.model_dir = os.path.join(output_dir, "models")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Save configuration
        self.config = {
            "base_model_name": base_model_name,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "train_ratio": train_ratio,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "alpha": alpha,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_ratio": clip_ratio,
            "max_grad_norm": max_grad_norm,
            "device": str(self.device),
            "seed": seed,
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        
        # Initialize agent
        self.agent = None
    
    def prepare_data(self) -> Dict[str, Any]:
        """
        Prepare and process data for training.
        
        Returns:
            Dictionary containing processed training and testing data
        """
        logger.info("Preparing data...")
        processed_data_dir = os.path.join(self.output_dir, "processed_data")
        
        # Check if processed data already exists
        if os.path.exists(os.path.join(processed_data_dir, "combined_train.csv")) and \
           os.path.exists(os.path.join(processed_data_dir, "combined_test.csv")):
            logger.info("Loading pre-processed data...")
            train_data = pd.read_csv(os.path.join(processed_data_dir, "combined_train.csv"))
            test_data = pd.read_csv(os.path.join(processed_data_dir, "combined_test.csv"))
            
            symbols = []
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".csv"):
                    symbols.append(filename.replace(".csv", ""))
            
            return {
                "train": train_data,
                "test": test_data,
                "symbols": symbols
            }
        
        # Process data if it doesn't exist
        logger.info("Processing raw data...")
        data = prepare_all_symbols_data(
            self.data_dir,
            processed_data_dir,
            train_ratio=self.train_ratio
        )
        
        return data
    
    def initialize_agent(self):
        """Initialize the LLM agent with the base model."""
        logger.info(f"Initializing agent with base model: {self.base_model_name}")
        
        # Initialize agent with the base LLM model
        self.agent = LLMAgent(
            model_name=self.base_model_name,
            device=self.device
        )
    
    def collect_experience(
        self, 
        env: TradingEnvironment, 
        n_episodes: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Collect experience by running the agent in the environment.
        
        Args:
            env: Trading environment instance
            n_episodes: Number of episodes to collect
            
        Returns:
            List of episode trajectories
        """
        trajectories = []
        
        for episode in range(n_episodes):
            logger.info(f"Collecting experience - Episode {episode+1}/{n_episodes}")
            
            state = env.reset()
            done = False
            trajectory = {
                "states": [],
                "actions": [],
                "action_probs": [],
                "rewards": [],
                "next_states": [],
                "dones": [],
                "values": []
            }
            
            while not done:
                # Get agent's action and action probabilities
                action, action_prob, value = self.agent.act(state, deterministic=False, return_value=True)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Save transition
                trajectory["states"].append(state)
                trajectory["actions"].append(action)
                trajectory["action_probs"].append(action_prob)
                trajectory["rewards"].append(reward)
                trajectory["next_states"].append(next_state)
                trajectory["dones"].append(done)
                trajectory["values"].append(value)
                
                # Update state
                state = next_state
            
            # Convert lists to numpy arrays
            for k, v in trajectory.items():
                trajectory[k] = np.array(v)
            
            # Calculate returns and advantages
            trajectory["returns"] = self._compute_returns(
                trajectory["rewards"], 
                trajectory["dones"], 
                self.gamma
            )
            
            trajectory["advantages"] = self._compute_gae(
                trajectory["rewards"],
                trajectory["values"],
                trajectory["dones"],
                self.gamma,
                self.gae_lambda
            )
            
            # Add trajectory info
            trajectory["total_reward"] = np.sum(trajectory["rewards"])
            trajectory["episode_length"] = len(trajectory["rewards"])
            
            trajectories.append(trajectory)
            
            logger.info(f"Episode {episode+1} - Total Reward: {trajectory['total_reward']:.2f}, "
                       f"Length: {trajectory['episode_length']}")
        
        return trajectories
    
    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
        """Compute discounted returns."""
        returns = np.zeros_like(rewards)
        next_return = 0
        
        for t in reversed(range(len(rewards))):
            next_return = rewards[t] + gamma * next_return * (1 - dones[t])
            returns[t] = next_return
            
        return returns
    
    def _compute_gae(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        dones: np.ndarray, 
        gamma: float, 
        lam: float
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
            
        return advantages
    
    def train_epoch(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train the agent for one epoch on collected trajectories.
        
        Args:
            trajectories: List of collected trajectories
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare data
        states = np.concatenate([t["states"] for t in trajectories])
        actions = np.concatenate([t["actions"] for t in trajectories])
        old_action_probs = np.concatenate([t["action_probs"] for t in trajectories])
        returns = np.concatenate([t["returns"] for t in trajectories])
        advantages = np.concatenate([t["advantages"] for t in trajectories])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset indices
        indices = np.arange(len(states))
        n_batches = len(indices) // self.batch_size
        
        # Training metrics
        metrics = {
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "total_loss": 0,
            "approx_kl": 0,
            "clip_fraction": 0,
        }
        
        # GRPO update
        for _ in range(1):  # Number of epochs to train on the same data
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_action_probs = old_action_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                new_action_probs, values, entropy = self.agent.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Calculate ratio of new and old action probabilities
                ratio = new_action_probs / (batch_old_action_probs + 1e-8)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                
                # Calculate total loss
                loss = policy_loss + 0.5 * value_loss - self.alpha * entropy.mean()
                
                # Optimize
                self.agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
                self.agent.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    log_ratio = torch.log(ratio)
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                
                # Update metrics
                metrics["policy_loss"] += policy_loss.item() / n_batches
                metrics["value_loss"] += value_loss.item() / n_batches
                metrics["entropy"] += entropy.mean().item() / n_batches
                metrics["total_loss"] += loss.item() / n_batches
                metrics["approx_kl"] += approx_kl / n_batches
                metrics["clip_fraction"] += clip_fraction / n_batches
        
        return metrics
    
    def evaluate(self, env: TradingEnvironment, n_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate the agent in the environment.
        
        Args:
            env: Trading environment instance
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        eval_rewards = []
        eval_lengths = []
        returns = []
        portfolio_values = []
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            episode_portfolio_values = [env.portfolio_value]
            
            while not done:
                # Get deterministic action
                action, _, _ = self.agent.act(state, deterministic=True, return_value=True)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                episode_portfolio_values.append(env.portfolio_value)
                
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(step_count)
            returns.append(info.get("total_return", 0))
            portfolio_values.append(episode_portfolio_values)
            
            logger.info(f"Eval Episode {episode+1} - Reward: {episode_reward:.2f}, "
                       f"Return: {info.get('total_return', 0):.2f}, "
                       f"Steps: {step_count}")
        
        # Calculate metrics
        metrics = {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_length": np.mean(eval_lengths),
            "std_length": np.std(eval_lengths),
            "sharpe_ratio": calculate_metrics(returns)["sharpe_ratio"],
            "max_drawdown": calculate_metrics(returns)["max_drawdown"],
        }
        
        # Plot portfolio values
        self._plot_portfolio_values(portfolio_values)
        
        return metrics
    
    def _plot_portfolio_values(self, portfolio_values: List[List[float]]):
        """Plot portfolio values for evaluation episodes."""
        plt.figure(figsize=(10, 6))
        for i, values in enumerate(portfolio_values):
            plt.plot(values, label=f"Episode {i+1}")
        
        plt.xlabel("Steps")
        plt.ylabel("Portfolio Value")
        plt.title("Portfolio Value During Evaluation")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        fig_path = os.path.join(self.log_dir, f"portfolio_values_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(fig_path)
        plt.close()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint and training metrics."""
        # Save model
        model_path = os.path.join(self.model_dir, f"model_epoch_{epoch}.pt")
        self.agent.save(model_path)
        
        # Save metrics
        metrics_path = os.path.join(self.log_dir, f"metrics_epoch_{epoch}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Saved checkpoint for epoch {epoch}")
    
    def train(
        self,
        n_epochs: int,
        n_train_episodes_per_epoch: int = 10,
        n_eval_episodes: int = 5,
        save_freq: int = 1,
        symbols: Optional[List[str]] = None,
    ):
        """
        Train the agent using GRPO.
        
        Args:
            n_epochs: Number of training epochs
            n_train_episodes_per_epoch: Number of episodes to collect per epoch
            n_eval_episodes: Number of episodes for evaluation
            save_freq: Frequency to save checkpoints
            symbols: List of symbols to train on (if None, use all available)
        """
        # Prepare data
        data = self.prepare_data()
        
        # Initialize agent if not already initialized
        if self.agent is None:
            self.initialize_agent()
        
        # Use specified symbols or all available
        if symbols is None:
            symbols = list(data["symbols"]) if isinstance(data["symbols"], dict) else data["symbols"]
        
        logger.info(f"Starting training with {len(symbols)} symbols for {n_epochs} epochs")
        
        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            
            # Randomly select a symbol for this epoch
            symbol = random.choice(symbols)
            logger.info(f"Training on symbol: {symbol}")
            
            # Create environment for training
            train_env = TradingEnvironment(
                data=data["symbols"][symbol]["train"] if isinstance(data["symbols"], dict) else data["train"],
                symbol=symbol,
                initial_balance=10000,
                commission=0.001,
            )
            
            # Collect experience
            trajectories = self.collect_experience(
                env=train_env,
                n_episodes=n_train_episodes_per_epoch
            )
            
            # Train on collected experience
            train_metrics = self.train_epoch(trajectories)
            
            logger.info(f"Train Metrics - Policy Loss: {train_metrics['policy_loss']:.4f}, "
                       f"Value Loss: {train_metrics['value_loss']:.4f}, "
                       f"Entropy: {train_metrics['entropy']:.4f}")
            
            # Log training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            
            # Create environment for evaluation
            eval_env = TradingEnvironment(
                data=data["symbols"][symbol]["test"] if isinstance(data["symbols"], dict) else data["test"],
                symbol=symbol,
                initial_balance=10000,
                commission=0.001,
            )
            
            # Evaluate agent
            eval_metrics = self.evaluate(
                env=eval_env,
                n_episodes=n_eval_episodes
            )
            
            logger.info(f"Eval Metrics - Mean Reward: {eval_metrics['mean_reward']:.2f}, "
                       f"Mean Return: {eval_metrics['mean_return']:.2f}, "
                       f"Sharpe: {eval_metrics['sharpe_ratio']:.2f}")
            
            # Log evaluation metrics
            for key, value in eval_metrics.items():
                self.writer.add_scalar(f"eval/{key}", value, epoch)
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                metrics = {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "eval": eval_metrics,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.save_checkpoint(epoch + 1, metrics)
        
        logger.info("Training completed")
        self.writer.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    trainer = GRPOTrainer(
        base_model_name="Qwen/QwQ-32B",
        data_dir="/Users/progyan/fortress/coinframe/data/BitMart/csv",
        output_dir="training_results",
        train_ratio=0.75,
        batch_size=32,
        learning_rate=1e-5,
    )
    
    trainer.train(
        n_epochs=10,
        n_train_episodes_per_epoch=5,
        n_eval_episodes=3,
        save_freq=1,
    )