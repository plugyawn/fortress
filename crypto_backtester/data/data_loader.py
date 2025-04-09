"""
Data loader for cryptocurrency price data from various sources.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class CryptoDataLoader:
    """Loads and processes cryptocurrency price data from various sources."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing price data files
        """
        self.data_dir = data_dir
        self._validate_data_dir()
        
    def _validate_data_dir(self) -> None:
        """Validate that the data directory exists and contains CSV files."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")
            
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
            
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        
    def _load_bitmart_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load and process a BitMart CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Convert time column to datetime
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Since we don't have OHLCV directly, we'll use ReferenceRateUSD as the close price
            # and create synthetic OHLCV data based on the reference rate
            df['close'] = df['ReferenceRateUSD']
            
            # Create synthetic OHLC data based on close price
            # This is a simplification - in a real scenario, you might want to use
            # more sophisticated methods to estimate OHLC from reference rates
            df['open'] = df['close'].shift(1)
            df['high'] = df['close'] * 1.001  # Assume 0.1% volatility
            df['low'] = df['close'] * 0.999   # Assume 0.1% volatility
            
            # Calculate volume based on market cap changes
            df['volume'] = df['CapMrktEstUSD'].diff().abs()
            
            # Drop unnecessary columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Forward fill any missing values
            df.fillna(method='ffill', inplace=True)
            
            # Drop any remaining NaN values
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
            
    def load_symbol_data(self, symbol: str, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a specific symbol.
        
        Args:
            symbol: Symbol to load (e.g., 'BTC', 'ETH')
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = os.path.join(self.data_dir, f"{symbol.lower()}.csv")
        if not os.path.exists(file_path):
            raise ValueError(f"No data file found for symbol {symbol}")
            
        df = self._load_bitmart_csv(file_path)
        
        # Filter by date range if specified
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
            
        return df
        
    def load_multiple_symbols(self, symbols: List[str],
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of symbols to load
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.load_symbol_data(symbol, start_date, end_date)
                data[symbol] = df
                logger.info(f"Loaded {len(df)} rows for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {str(e)}")
                continue
                
        return data
        
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in the data directory."""
        files = os.listdir(self.data_dir)
        symbols = [f.replace('.csv', '').upper() for f in files if f.endswith('.csv')]
        return sorted(symbols)
        
    def get_data_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """
        Get the date range for a symbol's data.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            Tuple of (start_date, end_date)
        """
        df = self.load_symbol_data(symbol)
        return df.index.min(), df.index.max() 