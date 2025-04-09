"""
LLM Agent for making trading decisions within the backtesting environment.
"""

import logging
from typing import Dict, Any, Tuple, Union

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from crypto_backtester.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class LLMAgent(BaseAgent):
    """Loads and interacts with a Large Language Model for trading decisions."""

    def __init__(self, model_name: str = "Qwen/QwQ-32B", device_map: str = "auto", torch_dtype: str = "auto"):
        """
        Initializes the LLM agent.

        Args:
            model_name (str): The name of the Hugging Face model to load (e.g., "Qwen/QwQ-32B").
            device_map (str): Device mapping strategy (e.g., "auto", "cuda:0").
            torch_dtype (str): Data type for model loading (e.g., "auto", "torch.bfloat16").
        """
        super().__init__()
        logger.info(f"Initializing LLM agent with model: {model_name}")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True
            )
            logger.info(f"Successfully loaded model {model_name}")
        except ImportError as e:
             logger.error(f"ImportError: {e}. Make sure you have transformers, torch, and potentially accelerate installed.")
             raise
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
            raise

    def _construct_prompt(self, state_summary: str, historical_data_summary: str) -> str:
        """
        Constructs a detailed prompt for the LLM based on environment state.
        
        Args:
            state_summary (str): Text summary of the current environment state from BacktestingEnv.
            historical_data_summary (str): Summary of recent price action and indicators (can be derived from observation).
        
        Returns:
            str: The formatted prompt string.
        """
        prompt = f"You are a cryptocurrency trading assistant. Your goal is to maximize risk-adjusted returns (Sortino ratio) by analyzing market data and portfolio status.

        Current Market & Portfolio State:
        {state_summary}

        Recent Market Data Summary:
        {historical_data_summary}
        
        # --- Future Integrations (Placeholder) ---
        # Live News Feed Summary: [No data available yet]
        # Knowledge Graph Insights: [No data available yet]
        # -------------------------------------

        Based on the provided information, decide the best trading action for the next step. Consider the current position, unrealized PnL, recent price trends, indicator values, and the goal of maximizing Sortino ratio.
        
        Available Actions:
        0: Hold (Maintain current position or stay flat)
        1: Buy (Enter a long position using all available cash if currently flat)
        2: Sell (Close the existing long position if currently holding one)
        
        Output format: Provide your reasoning first, then output the chosen action number within <action> tags.
        Example Reasoning: The RSI is overbought (75) and the price has hit resistance. Closing the position to lock in profit.
        Example Output: <action>2</action>"
        
        return prompt

    def _parse_action(self, response: str) -> int:
        """
        Parses the LLM's response to extract the chosen action.

        Args:
            response (str): The full text response from the LLM.

        Returns:
            int: The extracted action (0, 1, or 2), or 0 (Hold) if parsing fails.
        """
        try:
            action_str = response.split("<action>")[1].split("</action>")[0].strip()
            action = int(action_str)
            if action in [0, 1, 2]:
                return action
            else:
                logger.warning(f"LLM returned invalid action number: {action}. Defaulting to Hold (0).")
                return 0
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse action from LLM response: '{response}'. Error: {e}. Defaulting to Hold (0).")
            return 0

    def get_action(self, state_summary: str, historical_data_summary: str) -> tuple[int, str, str]:
        """
        Gets the next trading action from the LLM.

        Args:
            state_summary (str): Text summary of the current environment state.
            historical_data_summary (str): Summary of recent price action and indicators.

        Returns:
            tuple[int, str, str]: A tuple containing:
                - action (int): The chosen action (0, 1, or 2).
                - reasoning (str): The reasoning part of the LLM's response.
                - full_prompt (str): The full prompt sent to the LLM.
        """
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not loaded. Cannot get action.")
            return 0, "Error: Model not loaded", ""

        full_prompt = self._construct_prompt(state_summary, historical_data_summary)
        
        messages = [
            {"role": "user", "content": full_prompt}
        ]
        
        # Note: Qwen models often use apply_chat_template for formatting
        try:
            # Use apply_chat_template if available and appropriate for the model
            if hasattr(self.tokenizer, 'apply_chat_template'):
                 inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True, # Important for generation
                    return_tensors="pt"
                ).to(self.model.device)
            else:
                # Fallback for older tokenizer versions or models not using chat templates
                 inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

            # Generate response
            with torch.no_grad():
                 outputs = self.model.generate(
                    inputs,
                    max_new_tokens=200, # Adjust as needed
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                 )
            
            # Decode the response
            # Need to handle potential skipping of special tokens and prompt
            response_ids = outputs[0][inputs.shape[1]:] # Get only generated tokens
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            action = self._parse_action(response_text)
            reasoning = response_text.split("<action>")[0].strip() # Extract reasoning part
            
            logger.debug(f"LLM Prompt: {full_prompt}")
            logger.debug(f"LLM Response: {response_text}")
            logger.debug(f"Parsed Action: {action}, Reasoning: {reasoning}")
            
            return action, reasoning, full_prompt
        
        except Exception as e:
            logger.error(f"Error during LLM inference: {e}", exc_info=True)
            return 0, f"Error during inference: {e}", full_prompt # Default to Hold on error 
            
    def act(self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        """
        Implements the BaseAgent interface for action selection.
        
        Args:
            state: Current state observation (will be converted to a text summary internally)
            deterministic: Not used for LLM agent
            
        Returns:
            Tuple of (action, info_dict)
        """
        # Convert state tensor to text summaries
        # This is a simplified implementation - in practice, you would need to convert the
        # state tensor to the text summaries expected by the LLM
        state_summary = "Current portfolio value: $10,000, Position: None, Cash: $10,000"
        historical_summary = "Price: $50,000, 24h Change: +5%, RSI: 55, MACD: Bullish"
        
        action, reasoning, _ = self.get_action(state_summary, historical_summary)
        
        info = {
            "reasoning": reasoning,
            "action_type": ["hold", "buy", "sell"][action]
        }
        
        return action, info
        
    def reset(self):
        """Reset the agent's internal state if needed."""
        pass  # LLM agent is stateless, so nothing to reset