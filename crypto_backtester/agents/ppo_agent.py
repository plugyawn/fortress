"""
Proximal Policy Optimization (PPO) agent implementation for cryptocurrency trading.
This module implements a PPO agent with continuous action space for trading.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Dict, Union, List, Optional, Any

from crypto_backtester.agents.base_agent import BaseAgent


class ActorCritic(nn.Module):
    """
    Actor-Critic network for the PPO agent.
    Contains both the policy network (actor) and the value network (critic).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_actions: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize the ActorCritic network.
        
        Args:
            input_dim: Dimension of the input feature vector
            hidden_dim: Dimension of the hidden layers
            num_actions: Number of discrete actions (typically 3: buy, hold, sell)
            num_layers: Number of hidden layers
            dropout: Dropout rate
            activation: Activation function to use (relu, tanh, leaky_relu)
        """
        super().__init__()
        
        # Define activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Shared feature extractor
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(nn.Linear(input_dim, hidden_dim))
        self.shared_layers.append(self.activation)
        self.shared_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            self.shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.shared_layers.append(self.activation)
            self.shared_layers.append(nn.Dropout(dropout))
        
        # Actor (policy) head
        self.actor_head = nn.Linear(hidden_dim, num_actions)
        
        # Critic (value) head
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (action_distribution, value)
        """
        # Shared feature extraction
        for layer in self.shared_layers:
            x = layer(x)
        
        # Actor (policy) head
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = Categorical(action_probs)
        
        # Critic (value) head
        value = self.critic_head(x)
        
        return action_dist, value


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent implementation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_actions: int = 3,  # buy, hold, sell
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        learning_rate: float = 0.0001,
        **kwargs
    ):
        """
        Initialize the PPO agent.
        
        Args:
            input_dim: Dimension of the input feature vector
            hidden_dim: Dimension of the hidden layers
            num_actions: Number of discrete actions
            num_layers: Number of hidden layers
            dropout: Dropout rate
            activation: Activation function
            learning_rate: Learning rate for optimizer
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Create actor-critic network
        self.network = ActorCritic(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass through the agent's network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (action_distribution, value)
        """
        return self.network(x)
    
    def act(self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            deterministic: Whether to select the action deterministically (greedy) or sample from the distribution
            
        Returns:
            Tuple of (action, log_info)
        """
        # Convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        # Move to the same device as the model
        device = next(self.parameters()).device
        state = state.to(device)
        
        # Get action distribution and value
        with torch.no_grad():
            action_dist, value = self.forward(state)
            
            if deterministic:
                # Greedy action
                action = torch.argmax(action_dist.probs, dim=-1).item()
            else:
                # Sample from distribution
                action = action_dist.sample().item()
            
            # Get log probability of the action
            log_prob = action_dist.log_prob(torch.tensor([action], device=device)).item()
        
        # Return action and additional info
        info = {
            "value": value.item(),
            "log_prob": log_prob,
            "action_probs": action_dist.probs.cpu().numpy()
        }
        
        return action, info
    
    def reset(self):
        """Reset agent's internal state if needed."""
        pass  # PPO is a stateless agent, so nothing to reset
    
    def to(self, device):
        """Move agent to specified device."""
        self.network = self.network.to(device)
        return super().to(device)
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dictionary for the agent.
        Handles both directly loading to agent or to the internal network.
        """
        try:
            super().load_state_dict(state_dict, strict)
        except:
            self.network.load_state_dict(state_dict, strict)
        return self