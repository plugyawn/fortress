"""
Base agent class for all trading agents in the crypto backtester.
"""

import torch.nn as nn
from typing import Dict, Any, Tuple, Union
import numpy as np
import torch

class BaseAgent(nn.Module):
    """
    Base class for all trading agents.
    
    This class provides the interface that all agents should implement.
    It inherits from nn.Module to facilitate using PyTorch for neural network-based agents.
    """
    
    def __init__(self):
        """Initialize the base agent."""
        super().__init__()
    
    def act(self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            deterministic: Whether to select the action deterministically or stochastically
            
        Returns:
            Tuple of (action, info_dict)
        """
        raise NotImplementedError("Subclasses must implement the act method")
    
    def reset(self):
        """Reset the agent's internal state."""
        pass
    
    def get_action(self, *args, **kwargs):
        """
        Alternative interface for selecting an action.
        
        This method is provided for compatibility with agents that use a different interface,
        such as the LLMAgent.
        """
        raise NotImplementedError("Subclasses can implement the get_action method if needed")