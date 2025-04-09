"""
Technical feature extraction for cryptocurrency data.
This module contains functions to compute technical indicators from price data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any
import logging
import talib

logger = logging.getLogger(__name__)

def add_technical_features(
    df: pd.DataFrame, 
    features_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Add technical indicators to price dataframe.
    
    Args:
        df: DataFrame with columns 'open', 'high', 'low', 'close', 'volume'
        features_config: Configuration dictionary for feature extraction
            
    Returns:
        DataFrame with added technical indicators
    """
    if features_config is None:
        features_config = {
            "momentum_indicators": True,
            "volatility_indicators": True,
            "volume_indicators": True,
            "cycle_indicators": True,
            "pattern_recognition": False,  # These create a lot of binary flags
            "ma_types": ["SMA", "EMA", "WMA"],
            "ma_periods": [7, 14, 21, 50, 200],
            "rsi_periods": [7, 14, 21],
            "bbands_periods": [20],
        }
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            logger.error(f"Required column '{col}' not found in DataFrame")
            return df
    
    # Extract numpy arrays for TA-Lib
    open_prices = result_df['open'].values
    high_prices = result_df['high'].values
    low_prices = result_df['low'].values
    close_prices = result_df['close'].values
    volumes = result_df['volume'].values
    
    # Momentum Indicators
    if features_config.get("momentum_indicators", True):
        # RSI - Relative Strength Index
        for period in features_config.get("rsi_periods", [14]):
            result_df[f'RSI_{period}'] = talib.RSI(close_prices, timeperiod=period)
        
        # MACD - Moving Average Convergence Divergence
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        result_df['MACD'] = macd
        result_df['MACD_signal'] = macd_signal
        result_df['MACD_hist'] = macd_hist
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            high_prices, 
            low_prices, 
            close_prices, 
            fastk_period=14, 
            slowk_period=3, 
            slowk_matype=0, 
            slowd_period=3, 
            slowd_matype=0
        )
        result_df['STOCH_k'] = slowk
        result_df['STOCH_d'] = slowd
        
        # Rate of Change
        result_df['ROC_10'] = talib.ROC(close_prices, timeperiod=10)
        
        # Commodity Channel Index
        result_df['CCI_14'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
    
    # Volatility Indicators
    if features_config.get("volatility_indicators", True):
        # Bollinger Bands
        for period in features_config.get("bbands_periods", [20]):
            upperband, middleband, lowerband = talib.BBANDS(
                close_prices, 
                timeperiod=period, 
                nbdevup=2, 
                nbdevdn=2, 
                matype=0
            )
            result_df[f'BB_upper_{period}'] = upperband
            result_df[f'BB_middle_{period}'] = middleband
            result_df[f'BB_lower_{period}'] = lowerband
            # Calculate BB width
            result_df[f'BB_width_{period}'] = (upperband - lowerband) / middleband
        
        # Average True Range
        result_df['ATR_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Historical Volatility
        result_df['STDDEV_20'] = talib.STDDEV(close_prices, timeperiod=20, nbdev=1)
    
    # Volume Indicators
    if features_config.get("volume_indicators", True):
        # On-Balance Volume
        result_df['OBV'] = talib.OBV(close_prices, volumes)
        
        # Chaikin A/D Line
        result_df['AD'] = talib.AD(high_prices, low_prices, close_prices, volumes)
        
        # Chaikin A/D Oscillator
        result_df['ADOSC'] = talib.ADOSC(high_prices, low_prices, close_prices, volumes, fastperiod=3, slowperiod=10)
        
        # Money Flow Index
        result_df['MFI_14'] = talib.MFI(high_prices, low_prices, close_prices, volumes, timeperiod=14)
    
    # Moving Averages
    ma_types_map = {
        "SMA": talib.MA_Type.SMA,
        "EMA": talib.MA_Type.EMA,
        "WMA": talib.MA_Type.WMA,
        "DEMA": talib.MA_Type.DEMA,
        "TEMA": talib.MA_Type.TEMA,
        "TRIMA": talib.MA_Type.TRIMA,
        "KAMA": talib.MA_Type.KAMA,
        "MAMA": talib.MA_Type.MAMA,
        "T3": talib.MA_Type.T3
    }
    
    ma_types = features_config.get("ma_types", ["SMA", "EMA"])
    ma_periods = features_config.get("ma_periods", [7, 14, 21, 50, 200])
    
    for ma_type in ma_types:
        if ma_type in ma_types_map:
            for period in ma_periods:
                ma_func = getattr(talib, ma_type, None)
                if ma_func:
                    result_df[f'{ma_type}_{period}'] = ma_func(close_prices, timeperiod=period)
                else:
                    result_df[f'{ma_type}_{period}'] = talib.MA(close_prices, timeperiod=period, matype=ma_types_map[ma_type])
    
    # Compute price relative to moving averages
    for ma_type in ma_types:
        for period in ma_periods:
            ma_col = f'{ma_type}_{period}'
            if ma_col in result_df.columns:
                result_df[f'price_rel_{ma_col}'] = result_df['close'] / result_df[ma_col] - 1
    
    # Cycle Indicators
    if features_config.get("cycle_indicators", True):
        # Hilbert Transform
        result_df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_prices)
        result_df['HT_DCPHASE'] = talib.HT_DCPHASE(close_prices)
        result_df['HT_PHASOR_inphase'], result_df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close_prices)
        result_df['HT_SINE'], result_df['HT_LEADSINE'] = talib.HT_SINE(close_prices)
    
    # Pattern Recognition
    if features_config.get("pattern_recognition", False):
        # Candlestick patterns (generates many binary indicators)
        pattern_funcs = [
            ('CDL2CROWS', talib.CDL2CROWS),
            ('CDL3BLACKCROWS', talib.CDL3BLACKCROWS),
            ('CDL3INSIDE', talib.CDL3INSIDE),
            ('CDL3LINESTRIKE', talib.CDL3LINESTRIKE),
            ('CDL3OUTSIDE', talib.CDL3OUTSIDE),
            ('CDL3STARSINSOUTH', talib.CDL3STARSINSOUTH),
            ('CDL3WHITESOLDIERS', talib.CDL3WHITESOLDIERS),
            ('CDLABANDONEDBABY', talib.CDLABANDONEDBABY),
            ('CDLBELTHOLD', talib.CDLBELTHOLD),
            ('CDLBREAKAWAY', talib.CDLBREAKAWAY),
            ('CDLCLOSINGMARUBOZU', talib.CDLCLOSINGMARUBOZU)
        ]
        
        for pattern_name, pattern_func in pattern_funcs:
            result_df[pattern_name] = pattern_func(open_prices, high_prices, low_prices, close_prices)
    
    # Add log returns
    result_df['log_return'] = np.log(close_prices / np.roll(close_prices, 1))
    result_df.loc[0, 'log_return'] = 0  # Replace NaN for first row
    
    # Fill NaN values
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df = result_df.fillna(method='ffill').fillna(0)
    
    logger.info(f"Added {len(result_df.columns) - len(df.columns)} technical features")
    
    return result_df

def prepare_all_symbols_data(
    data_dir: str,
    output_dir: str,
    symbols: Optional[List[str]] = None,
    time_range: Optional[Dict[str, str]] = None,
    features_config: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Prepare data for all symbols in data_dir by adding technical features.
    
    Args:
        data_dir: Directory containing CSV files with price data
        output_dir: Directory to save processed data files
        symbols: List of symbols to process (None = process all in data_dir)
        time_range: Dict with 'start' and 'end' dates to filter data
        features_config: Configuration for technical features
        
    Returns:
        Dictionary mapping symbol to output file path
    """
    import os
    import glob
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in data_dir
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # Process only specified symbols if provided
    if symbols:
        csv_files = [f for f in csv_files if any(s in os.path.basename(f) for s in symbols)]
    
    results = {}
    for csv_file in csv_files:
        symbol = os.path.basename(csv_file).split('.')[0]
        logger.info(f"Processing data for {symbol}")
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_file)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.set_index('timestamp')
            
            # Filter by time range if provided
            if time_range:
                start = pd.to_datetime(time_range.get('start'))
                end = pd.to_datetime(time_range.get('end'))
                if start:
                    df = df[df.index >= start]
                if end:
                    df = df[df.index <= end]
            
            # Add technical features
            df_with_features = add_technical_features(df, features_config)
            
            # Save processed data
            output_file = os.path.join(output_dir, f"{symbol}_processed.csv")
            df_with_features.to_csv(output_file)
            results[symbol] = output_file
            
            logger.info(f"Saved processed data for {symbol} to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    return results