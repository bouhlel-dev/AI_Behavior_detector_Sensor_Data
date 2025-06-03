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
    parser = argparse.ArgumentParser(description='Generate predictions for BFRB detection')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Directory to save predictions')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    
    # Preprocessing parameters
    parser.add_argument('--window_size', type=int, default=1000,
                        help='Window size in milliseconds')
    parser.add_argument('--step_size', type=int, default=500,
                        help='Step size for sliding window in milliseconds')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalize the data')
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths to data files
    test_data_path = os.path.join(args.data_dir, 'test_series.parquet')
    sample_submission_path = os.path.join(args.data_dir, 'sample_submission.csv')
    
    # Initialize preprocessor
    preprocessor = TimeSeriesPreprocessor(
        window_size=args.window_size,
        step_size=args.step_size,
        normalize=args.normalize
    )
    
    # Preprocess test data
    logger.info("Preprocessing test data...")
    features, series_ids = preprocessor.preprocess(
        data_path=test_data_path,
        labels_path=None  # No labels for test data
    )
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = BFRBModel()
    model.load_model(args.model_path)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(features)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'series_id': series_ids,
        'prediction': predictions
    })
    
    # If multiple predictions per series_id, take the most common prediction
    submission_df = submission_df.groupby('series_id')['prediction'].agg(
        lambda x: pd.Series.mode(x)[0]
    ).reset_index()
    
    # Ensure all series_ids from sample submission are included
    sample_submission = pd.read_csv(sample_submission_path)
    all_series_ids = set(sample_submission['series_id'])
    predicted_series_ids = set(submission_df['series_id'])
    
    # Check for missing series_ids
    missing_series_ids = all_series_ids - predicted_series_ids
    if missing_series_ids:
        logger.warning(f"Missing predictions for {len(missing_series_ids)} series_ids")
        # For missing series_ids, predict the most common class
        most_common_class = pd.Series.mode(predictions)[0]
        missing_df = pd.DataFrame({
            'series_id': list(missing_series_ids),
            'prediction': [most_common_class] * len(missing_series_ids)
        })
        submission_df = pd.concat([submission_df, missing_df], ignore_index=True)
    
    # Save submission file
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()