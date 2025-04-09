"""
Data preparation utilities for processing CSV files and preparing train/test splits.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def parse_csv_file(filepath: str) -> pd.DataFrame:
    """
    Parse a cryptocurrency CSV file and standardize the format.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with standardized OHLCV format
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Check minimum required columns
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not all(col.lower() in map(str.lower, df.columns) for col in required_cols):
            # Try common alternatives for column naming
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Standardize column names
            rename_map = {}
            for col in df.columns:
                if 'open' in col.lower():
                    rename_map[col] = 'open'
                elif 'high' in col.lower():
                    rename_map[col] = 'high'
                elif 'low' in col.lower():
                    rename_map[col] = 'low'
                elif 'close' in col.lower():
                    rename_map[col] = 'close'
                elif 'volume' in col.lower():
                    rename_map[col] = 'volume'
            
            df = df.rename(columns=rename_map)
            
            # Check again
            if not all(col in df.columns for col in required_cols):
                logger.error(f"File {filepath} doesn't have required OHLCV columns after renaming")
                return pd.DataFrame()
        
        # Ensure numeric types for OHLCV columns
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna(subset=required_cols)
        
        # Sort by index if it's a timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        return pd.DataFrame()

def prepare_data_directory(csv_dir: str, output_dir: str, train_ratio: float = 0.75) -> Dict[str, Dict[str, str]]:
    """
    Process all CSV files in a directory and split into train/test datasets.
    
    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory to save processed files
        train_ratio: Ratio of data to use for training (default: 0.75)
        
    Returns:
        Dictionary of file paths for each symbol's train and test sets
    """
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    result = {}
    
    # Process each CSV file
    for filename in os.listdir(csv_dir):
        if not filename.endswith('.csv'):
            continue
        
        symbol = filename.replace('.csv', '')
        filepath = os.path.join(csv_dir, filename)
        
        logger.info(f"Processing {symbol} from {filepath}")
        
        # Parse CSV
        df = parse_csv_file(filepath)
        if df.empty:
            logger.warning(f"Skipping empty or invalid file: {filepath}")
            continue
        
        # Split into train/test
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Save processed files
        train_path = os.path.join(train_dir, f"{symbol}.csv")
        test_path = os.path.join(test_dir, f"{symbol}.csv")
        
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        
        result[symbol] = {
            'train': train_path,
            'test': test_path
        }
        
        logger.info(f"Saved {symbol} - Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    return result

def extract_features(df: pd.DataFrame, lookback_window: int = 30) -> pd.DataFrame:
    """
    Extract features from OHLCV data for machine learning.
    
    Args:
        df: DataFrame with OHLCV data
        lookback_window: Number of periods to look back for rolling features
        
    Returns:
        DataFrame with extracted features
    """
    # Ensure required columns exist
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
        return df
    
    # Make a copy to avoid modifying the original
    feature_df = df.copy()
    
    # Price-based features
    feature_df['returns'] = feature_df['close'].pct_change()
    feature_df['log_returns'] = np.log(feature_df['close'] / feature_df['close'].shift(1))
    
    # Rolling window features
    feature_df['rolling_mean'] = feature_df['close'].rolling(window=lookback_window).mean()
    feature_df['rolling_std'] = feature_df['close'].rolling(window=lookback_window).std()
    feature_df['upper_band'] = feature_df['rolling_mean'] + (feature_df['rolling_std'] * 2)
    feature_df['lower_band'] = feature_df['rolling_mean'] - (feature_df['rolling_std'] * 2)
    
    # Relative measures
    feature_df['dist_to_upper'] = (feature_df['upper_band'] - feature_df['close']) / feature_df['close']
    feature_df['dist_to_lower'] = (feature_df['close'] - feature_df['lower_band']) / feature_df['close']
    
    # Volume features
    feature_df['volume_change'] = feature_df['volume'].pct_change()
    feature_df['volume_ma'] = feature_df['volume'].rolling(window=lookback_window).mean()
    feature_df['relative_volume'] = feature_df['volume'] / feature_df['volume_ma']
    
    # Volatility measures
    feature_df['atr'] = (
        feature_df['high'] - feature_df['low']
    ).rolling(window=lookback_window).mean()
    feature_df['atr_pct'] = feature_df['atr'] / feature_df['close']
    
    # Momentum indicators
    feature_df['rsi'] = calculate_rsi(feature_df['close'], periods=14)
    
    # Drop NaN values created by rolling windows
    feature_df = feature_df.dropna()
    
    return feature_df

def calculate_rsi(series: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def prepare_all_symbols_data(
    csv_dir: str, 
    output_dir: str, 
    min_rows: int = 1000, 
    train_ratio: float = 0.75
) -> Dict[str, pd.DataFrame]:
    """
    Process all symbols and prepare consolidated training and testing datasets.
    
    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory to save processed files
        min_rows: Minimum number of rows to consider a symbol valid
        train_ratio: Ratio of data to use for training
        
    Returns:
        Dictionary with train and test data frames
    """
    symbols_data = {}
    all_train_dfs = []
    all_test_dfs = []
    
    # Process each CSV file
    for filename in os.listdir(csv_dir):
        if not filename.endswith('.csv'):
            continue
        
        symbol = filename.replace('.csv', '')
        filepath = os.path.join(csv_dir, filename)
        
        try:
            logger.info(f"Processing {symbol} from {filepath}")
            
            # Parse CSV
            df = parse_csv_file(filepath)
            if df.empty:
                logger.warning(f"Skipping empty or invalid file: {filepath}")
                continue
            
            # Check minimum data requirement
            if len(df) < min_rows:
                logger.warning(f"Skipping {symbol}: insufficient data ({len(df)} < {min_rows})")
                continue
            
            # Extract features
            feature_df = extract_features(df)
            if len(feature_df) < min_rows - 100:  # Allow some loss due to feature calculation
                logger.warning(f"Skipping {symbol}: insufficient data after feature extraction")
                continue
            
            # Add symbol column
            feature_df['symbol'] = symbol
            
            # Split into train/test
            split_idx = int(len(feature_df) * train_ratio)
            train_df = feature_df.iloc[:split_idx]
            test_df = feature_df.iloc[split_idx:]
            
            all_train_dfs.append(train_df)
            all_test_dfs.append(test_df)
            
            symbols_data[symbol] = {
                'train': train_df,
                'test': test_df
            }
            
            logger.info(f"Processed {symbol} - Train: {len(train_df)} rows, Test: {len(test_df)} rows")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Combine all training and testing data
    combined_train = pd.concat(all_train_dfs) if all_train_dfs else pd.DataFrame()
    combined_test = pd.concat(all_test_dfs) if all_test_dfs else pd.DataFrame()
    
    # Save combined datasets
    os.makedirs(output_dir, exist_ok=True)
    if not combined_train.empty:
        combined_train.to_csv(os.path.join(output_dir, 'combined_train.csv'))
    if not combined_test.empty:
        combined_test.to_csv(os.path.join(output_dir, 'combined_test.csv'))
    
    return {
        'train': combined_train,
        'test': combined_test,
        'symbols': symbols_data
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    csv_dir = "/Users/progyan/fortress/coinframe/data/BitMart/csv"
    output_dir = "processed_data"
    
    result = prepare_all_symbols_data(csv_dir, output_dir)
    logger.info(f"Processed {len(result['symbols'])} symbols")
    logger.info(f"Combined train set: {len(result['train'])} rows")
    logger.info(f"Combined test set: {len(result['test'])} rows")