#!/usr/bin/env python3
"""
Training script for GRPO (Gradient-based Reward Policy Optimization) with detailed logging.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import json
import torch
from typing import Dict, Any

from crypto_backtester.training.grpo_trainer import TrainingConfig, create_default_training_pipeline
from crypto_backtester.agents.ppo_agent import PPOAgent
from crypto_backtester.data.data_loader import CryptoDataLoader

# Configure logging
def setup_logging(log_dir: str) -> None:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters and handlers
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log system info
    logging.info(f"Python version: {sys.version}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a trading agent using GRPO')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing price data CSV files')
    parser.add_argument('--symbols', type=str, nargs='+', required=True,
                        help='Symbols to train on')
    parser.add_argument('--train_start', type=str, required=True,
                        help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, required=True,
                        help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, required=True,
                        help='Testing start date (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, required=True,
                        help='Testing end date (YYYY-MM-DD)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='ppo',
                        choices=['ppo'], help='Type of model to train')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay for optimizer')
    
    # Environment parameters
    parser.add_argument('--window_size', type=int, default=50,
                        help='Observation window size')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='Initial balance for trading')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Trading commission rate')
    
    # Logging parameters
    parser.add_argument('--output_dir', type=str, default='training_output',
                        help='Directory for outputs (models, logs)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='crypto-trading',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this training run')
    
    return parser.parse_args()

def create_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create training configuration from command line arguments."""
    config_overrides = {
        "data_dir": args.data_dir,
        "symbols": args.symbols,
        "train_start_date": args.train_start,
        "train_end_date": args.train_end,
        "test_start_date": args.test_start,
        "test_end_date": args.test_end,
        "model_type": args.model_type,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "window_size": args.window_size,
        "initial_balance": args.initial_balance,
        "commission": args.commission,
        "model_dir": os.path.join(args.output_dir, "models"),
        "log_dir": os.path.join(args.output_dir, "logs"),
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "run_name": args.run_name or f"grpo_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    return config_overrides

def validate_data_availability(data_loader: CryptoDataLoader, symbols: list, 
                             train_start: str, train_end: str,
                             test_start: str, test_end: str) -> None:
    """Validate that data is available for the specified symbols and date ranges."""
    available_symbols = data_loader.get_available_symbols()
    missing_symbols = [s for s in symbols if s.upper() not in available_symbols]
    if missing_symbols:
        raise ValueError(f"Data not available for symbols: {missing_symbols}")
        
    for symbol in symbols:
        start_date, end_date = data_loader.get_data_range(symbol)
        train_start_dt = datetime.strptime(train_start, "%Y-%m-%d")
        train_end_dt = datetime.strptime(train_end, "%Y-%m-%d")
        test_start_dt = datetime.strptime(test_start, "%Y-%m-%d")
        test_end_dt = datetime.strptime(test_end, "%Y-%m-%d")
        
        if train_start_dt < start_date:
            raise ValueError(f"Training start date {train_start} is before available data for {symbol}")
        if test_end_dt > end_date:
            raise ValueError(f"Test end date {test_end} is after available data for {symbol}")
            
        logging.info(f"Data range for {symbol}: {start_date} to {end_date}")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(log_dir)
    logging.info("Starting training script")
    logging.info(f"Arguments: {args}")
    
    try:
        # Initialize data loader
        data_loader = CryptoDataLoader(args.data_dir)
        
        # Validate data availability
        validate_data_availability(
            data_loader, 
            args.symbols,
            args.train_start,
            args.train_end,
            args.test_start,
            args.test_end
        )
        
        # Create configuration
        config_overrides = create_config_from_args(args)
        
        # Save configuration
        config_path = os.path.join(args.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_overrides, f, indent=4)
        logging.info(f"Saved configuration to {config_path}")
        
        # Create training pipeline
        trainer, agent = create_default_training_pipeline(
            data_dir=args.data_dir,
            symbols=args.symbols,
            train_start_date=args.train_start,
            train_end_date=args.train_end,
            test_start_date=args.test_start,
            test_end_date=args.test_end,
            model_type=args.model_type,
            config_overrides=config_overrides
        )
        
        # Calculate total training steps
        # This is a rough estimate based on data size and epochs
        sample_df = data_loader.load_symbol_data(args.symbols[0], args.train_start, args.train_end)
        steps_per_epoch = len(sample_df) - args.window_size
        total_steps = steps_per_epoch * args.epochs * len(args.symbols)
        
        logging.info(f"Starting training for {total_steps} total steps")
        logging.info(f"Training will run for {args.epochs} epochs")
        logging.info(f"Using device: {next(agent.parameters()).device}")
        
        # Train the agent
        train_stats = trainer.train(total_steps)
        
        # Save final results
        results_path = os.path.join(args.output_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(train_stats, f, indent=4)
        logging.info(f"Saved training results to {results_path}")
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 