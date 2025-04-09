"""
Performance metrics calculation utilities
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate_period: float, period: int) -> float:
    """Calculates Sortino Ratio for a given period."""
    if len(returns) < period: 
        return 0.0

    relevant_returns = returns.iloc[-period:]
    mean_return = relevant_returns.mean()
    
    # Calculate downside deviation
    downside_returns = relevant_returns[relevant_returns < risk_free_rate_period]
    if len(downside_returns) <= 1: 
        return 0.0  # Need >1 point to calculate std dev
    
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns - risk_free_rate_period)))
    if downside_deviation < 1e-8: 
        return 0.0  # Avoid division by zero

    sortino = (mean_return - risk_free_rate_period) / downside_deviation
    return sortino if not np.isnan(sortino) else 0.0

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate_period: float, period: int) -> float:
    """Calculates Sharpe Ratio for a given period."""
    if len(returns) < period: 
        return 0.0

    relevant_returns = returns.iloc[-period:]
    mean_return = relevant_returns.mean()
    std_dev = relevant_returns.std()
    
    if std_dev < 1e-8 or len(relevant_returns) <= 1:
        return 0.0

    sharpe = (mean_return - risk_free_rate_period) / std_dev
    return sharpe if not np.isnan(sharpe) else 0.0

def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculates the maximum drawdown from a series of portfolio values."""
    if len(portfolio_values) < 2: 
        return 0.0
    
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() # Minimum value represents the largest drawdown percentage
    return abs(max_drawdown) if not np.isnan(max_drawdown) else 0.0

def calculate_performance_metrics(
    portfolio_history: List[Tuple[pd.Timestamp, float]],
    trade_history: List[Dict],
    initial_capital: float,
    risk_free_rate_annual: float,
    periods_per_year: float,
    total_steps: int
) -> Dict[str, Any]:
    """Calculates comprehensive performance metrics."""
    if not portfolio_history: 
        return {"Error": "No portfolio history available."}

    portfolio_df = pd.DataFrame(portfolio_history, columns=['Timestamp', 'Value']).set_index('Timestamp')
    if len(portfolio_df) < 2: 
        return {"Error": "Portfolio history too short."}

    final_value = portfolio_df['Value'].iloc[-1]
    total_return_perc = (final_value / initial_capital - 1) * 100

    # Calculate returns for risk metrics
    returns = portfolio_df['Value'].pct_change().dropna()
    if len(returns) < 2: 
        return {"Warning": "Not enough returns data for risk metrics."}

    risk_free_rate_period = (1 + risk_free_rate_annual)**(1/periods_per_year) - 1

    # Sharpe Ratio (using all returns)
    mean_return_period = returns.mean()
    std_dev_period = returns.std()
    sharpe_ratio = 0.0
    if std_dev_period > 1e-8:
        sharpe_ratio = (mean_return_period - risk_free_rate_period) / std_dev_period * np.sqrt(periods_per_year) # Annualize
        sharpe_ratio = sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0

    # Sortino Ratio (using all returns)
    downside_returns = returns[returns < risk_free_rate_period]
    sortino_ratio = 0.0
    if len(downside_returns) > 1:
        downside_deviation = np.sqrt(np.mean(np.square(downside_returns - risk_free_rate_period)))
        if downside_deviation > 1e-8:
            sortino_ratio = (mean_return_period - risk_free_rate_period) / downside_deviation * np.sqrt(periods_per_year) # Annualize
            sortino_ratio = sortino_ratio if not np.isnan(sortino_ratio) else 0.0

    # Max Drawdown
    max_drawdown_perc = calculate_max_drawdown(portfolio_df['Value']) * 100

    # Trade Stats
    closing_trades = [t for t in trade_history if t['type'] == 'Sell']
    num_trades = len(closing_trades)
    wins = [t for t in closing_trades if t.get('profit', 0) > 0]
    losses = [t for t in closing_trades if t.get('profit', 0) <= 0]
    win_rate = len(wins) / num_trades * 100 if num_trades > 0 else 0
    total_profit = sum(t['profit'] for t in wins)
    total_loss = abs(sum(t['profit'] for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    avg_profit_per_win = total_profit / len(wins) if wins else 0
    avg_loss_per_loss = total_loss / len(losses) if losses else 0

    return {
        "Initial Capital": initial_capital,
        "Final Portfolio Value": final_value,
        "Total Return (%)": total_return_perc,
        "Max Drawdown (%)": max_drawdown_perc,
        "Annualized Sharpe Ratio": sharpe_ratio,
        "Annualized Sortino Ratio": sortino_ratio,
        "Total Steps": total_steps,
        "Number of Trades": num_trades,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Average Winning Trade ($)": avg_profit_per_win,
        "Average Losing Trade ($)": avg_loss_per_loss,
        "Total Profit ($)": total_profit,
        "Total Loss ($)": total_loss,
        "Total Commissions ($)": sum(t['commission'] for t in trade_history),
    } 