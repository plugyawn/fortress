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

from fortress.agents.base_agent import BaseAgent


# ... rest of file 