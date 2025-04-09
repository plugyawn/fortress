"""
Technical indicator calculation utilities
"""

import logging
from typing import Dict, List

import pandas as pd
import pandas_ta as ta

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_indicators(df: pd.DataFrame, indicators_config: List[Dict]) -> pd.DataFrame:
    """Adds technical indicators using pandas_ta."""
    if df.empty: 
        return df
    
    logger.info(f"Calculating {len(indicators_config)} indicator group(s)...")

    # Ensure OHLCV are present and lowercase
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    strategy = ta.Strategy(name="Trading_Strategy", ta=indicators_config)
    df.ta.strategy(strategy)

    # Add time-based features
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour

    initial_len = len(df)
    df.dropna(inplace=True) # Drop rows with NaNs from indicator calculation
    logger.info(f"Dropped {initial_len - len(df)} rows with NaNs after indicator calculation.")
    return df 