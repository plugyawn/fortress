"""
Data fetching utilities for cryptocurrency data
"""

import os
import hashlib
import logging
from typing import Dict

import pandas as pd
from binance.client import Client

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

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