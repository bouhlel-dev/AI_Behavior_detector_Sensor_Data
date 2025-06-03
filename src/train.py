import os
import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from preprocess import TimeSeriesPreprocessor
from model import BFRBModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BFRB detection model')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Directory to save model and results')
    
    # Preprocessing parameters
    parser.add_argument('--window_size', type=int, default=1000,
                        help='Window size in milliseconds')
    parser.add_argument('--step_size', type=int, default=500,
                        help='Step size for sliding window in milliseconds')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalize the data')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='rf',
                        choices=['rf', 'xgb', 'gb', 'lr', 'stacked'],
                        help='Type of model to use (rf=RandomForest, xgb=XGBoost, gb=GradientBoosting, lr=LogisticRegression, stacked=StackedEnsemble)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds (0 for no CV)')
    parser.add_argument('--class_weight', action='store_true', default=True,
                        help='Whether to use class weights')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths to data files
    train_data_path = os.path.join(args.data_dir, 'train_series.parquet')
    train_labels_path = os.path.join(args.data_dir, 'train_labels.csv')
    
    # Initialize preprocessor
    preprocessor = TimeSeriesPreprocessor(
        window_size=args.window_size,
        step_size=args.step_size,
        normalize=args.normalize
    )
    
    # Preprocess data
    logger.info("Preprocessing data...")
    features, labels, series_ids = preprocessor.preprocess(
        data_path=train_data_path,
        labels_path=train_labels_path
    )
    
    # Calculate class weights if needed
    class_weights = None
    if args.class_weight:
        # Count labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        # Calculate weights inversely proportional to class frequencies
        weights = len(labels) / (len(unique_labels) * counts)
        class_weights = {label: weight for label, weight in zip(unique_labels, weights)}
        logger.info(f"Class weights: {class_weights}")
    
    # Initialize model
    model = BFRBModel(
        model_type=args.model_type,
        class_weights=class_weights,
        random_state=args.random_state
    )
    
    # Train model
    logger.info(f"Training {args.model_type} model...")
    cv = args.cv if args.cv > 0 else None
    model.train(features, labels, series_ids=series_ids, cv=cv)
    
    # Save model
    model_path = os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Get feature importance
    if hasattr(model.model, 'feature_importances_'):
        importance_df = model.get_feature_importance()
        importance_path = os.path.join(args.output_dir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")


if __name__ == "__main__":
    main()