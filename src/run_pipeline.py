import os
import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from preprocess import TimeSeriesPreprocessor
from model import BFRBModel
from utils import plot_confusion_matrix, plot_feature_importance, plot_class_distribution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run full BFRB detection pipeline')
    
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
    parser.add_argument('--model_type', type=str, default='stacked',
                        choices=['rf', 'xgb', 'gb', 'lr', 'stacked'],
                        help='Type of model to use (rf=RandomForest, xgb=XGBoost, gb=GradientBoosting, lr=LogisticRegression, stacked=StackedEnsemble)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds (0 for no CV)')
    parser.add_argument('--class_weight', action='store_true', default=True,
                        help='Whether to use class weights')
    
    # Run mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'full', 'compare'],
                        help='Mode to run (train, predict, full pipeline, or compare sensors)')
    
    # Sensor selection
    parser.add_argument('--imu_only', action='store_true',
                        help='Use only IMU data (no thermopile or ToF)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model (required for predict mode)')
    
    return parser.parse_args()


def run_training(args):
    """Run the training pipeline."""
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths to data files
    train_data_path = os.path.join(args.data_dir, 'train_series.parquet')
    train_labels_path = os.path.join(args.data_dir, 'train_labels.csv')
    
    # Load labels for visualization
    labels_df = pd.read_csv(train_labels_path)
    plot_class_distribution(labels_df)
    
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
    
    # Get feature importance if available
    if hasattr(model.model, 'feature_importances_'):
        # Create feature names (simplified for demonstration)
        feature_names = [f"feature_{i}" for i in range(len(model.model.feature_importances_))]
        importance_df = plot_feature_importance(model.model.feature_importances_, feature_names)
        importance_path = os.path.join(args.output_dir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
    
    return model, model_path


def run_prediction(args, model_path=None):
    """Run the prediction pipeline."""
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided model path or default
    if model_path is None:
        model_path = args.model_path
        if model_path is None:
            model_path = os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
            if not os.path.exists(model_path):
                raise ValueError(f"Model path not provided and default model {model_path} not found")
    
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
    logger.info(f"Loading model from {model_path}")
    model = BFRBModel()
    model.load_model(model_path)
    
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
    
    return submission_df


def filter_sensor_columns(data, imu_only=False):
    """Filter sensor columns based on sensor selection."""
    # Keep series_id and timestamp columns
    base_cols = ['series_id', 'timestamp']
    
    if imu_only:
        # Keep only IMU columns (accelerometer and gyroscope)
        sensor_cols = [col for col in data.columns if 'acc_' in col or 'rot_' in col]
    else:
        # Keep all sensor columns
        sensor_cols = [col for col in data.columns if col not in base_cols]
    
    # Return filtered dataframe
    return data[base_cols + sensor_cols]


def compare_sensors(args):
    """Compare model performance with and without thermopile/ToF sensors."""
    logger.info("Comparing performance with and without thermopile/ToF sensors")
    
    # Paths to data files
    train_data_path = os.path.join(args.data_dir, 'train_series.parquet')
    train_labels_path = os.path.join(args.data_dir, 'train_labels.csv')
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    preprocessor = TimeSeriesPreprocessor(
        window_size=args.window_size,
        step_size=args.step_size,
        normalize=args.normalize
    )
    
    # Load full data
    full_data = preprocessor.load_data(train_data_path)
    labels = preprocessor.load_labels(train_labels_path)
    
    # Create IMU-only data
    imu_data = filter_sensor_columns(full_data, imu_only=True)
    
    # Process both datasets
    results = {}
    
    # Process full sensor data
    logger.info("Processing full sensor data (IMU + Thermopile + ToF)")
    full_preprocessor = TimeSeriesPreprocessor(
        window_size=args.window_size,
        step_size=args.step_size,
        normalize=args.normalize
    )
    full_features, full_labels, full_series_ids = full_preprocessor.preprocess(
        data_path=train_data_path,
        labels_path=train_labels_path
    )
    
    # Process IMU-only data
    logger.info("Processing IMU-only data")
    # Save IMU-only data to temporary file
    imu_data_path = os.path.join(args.output_dir, 'imu_only_data.parquet')
    imu_data.to_parquet(imu_data_path)
    
    imu_preprocessor = TimeSeriesPreprocessor(
        window_size=args.window_size,
        step_size=args.step_size,
        normalize=args.normalize
    )
    imu_features, imu_labels, imu_series_ids = imu_preprocessor.preprocess(
        data_path=imu_data_path,
        labels_path=train_labels_path
    )
    
    # Train and evaluate models
    # Full sensor model
    logger.info(f"Training {args.model_type} model on full sensor data")
    full_model = BFRBModel(
        model_type=args.model_type,
        random_state=args.random_state
    )
    full_model.train(full_features, full_labels, series_ids=full_series_ids, cv=args.cv)
    
    # IMU-only model
    logger.info(f"Training {args.model_type} model on IMU-only data")
    imu_model = BFRBModel(
        model_type=args.model_type,
        random_state=args.random_state
    )
    imu_model.train(imu_features, imu_labels, series_ids=imu_series_ids, cv=args.cv)
    
    # Evaluate models
    logger.info("Evaluating full sensor model")
    y_pred_full = full_model.predict(full_features)
    full_f1_macro = f1_score(full_labels, y_pred_full, average='macro')
    
    # Binary F1 (BFRB vs non-BFRB)
    full_labels_binary = (full_labels > 0).astype(int)
    y_pred_full_binary = (y_pred_full > 0).astype(int)
    full_f1_binary = f1_score(full_labels_binary, y_pred_full_binary)
    
    # Combined score
    full_combined = (full_f1_macro + full_f1_binary) / 2
    
    logger.info("Evaluating IMU-only model")
    y_pred_imu = imu_model.predict(imu_features)
    imu_f1_macro = f1_score(imu_labels, y_pred_imu, average='macro')
    
    # Binary F1 (BFRB vs non-BFRB)
    imu_labels_binary = (imu_labels > 0).astype(int)
    y_pred_imu_binary = (y_pred_imu > 0).astype(int)
    imu_f1_binary = f1_score(imu_labels_binary, y_pred_imu_binary)
    
    # Combined score
    imu_combined = (imu_f1_macro + imu_f1_binary) / 2
    
    # Compare results
    logger.info("\nPerformance Comparison:")
    logger.info(f"Full Sensor Model - Binary F1: {full_f1_binary:.4f}, Macro F1: {full_f1_macro:.4f}, Combined: {full_combined:.4f}")
    logger.info(f"IMU-only Model - Binary F1: {imu_f1_binary:.4f}, Macro F1: {imu_f1_macro:.4f}, Combined: {imu_combined:.4f}")
    
    # Calculate improvement
    improvement = full_combined - imu_combined
    percent_improvement = 100 * improvement / imu_combined
    logger.info(f"Absolute Improvement: {improvement:.4f}")
    logger.info(f"Relative Improvement: {percent_improvement:.2f}%")
    
    # Save models
    full_model_path = os.path.join(args.output_dir, f"full_sensor_{args.model_type}_model.pkl")
    imu_model_path = os.path.join(args.output_dir, f"imu_only_{args.model_type}_model.pkl")
    full_model.save_model(full_model_path)
    imu_model.save_model(imu_model_path)
    logger.info(f"Models saved to {args.output_dir}")
    
    # Plot feature importance comparison if available
    if hasattr(full_model.model, 'feature_importances_'):
        full_importance = full_model.get_feature_importance()
        imu_importance = imu_model.get_feature_importance()
        
        # Save feature importance
        full_importance.to_csv(os.path.join(args.output_dir, "full_sensor_feature_importance.csv"), index=False)
        imu_importance.to_csv(os.path.join(args.output_dir, "imu_only_feature_importance.csv"), index=False)
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    metrics = ['Binary F1', 'Macro F1', 'Combined']
    imu_values = [imu_f1_binary, imu_f1_macro, imu_combined]
    full_values = [full_f1_binary, full_f1_macro, full_combined]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, imu_values, width, label='IMU Only')
    plt.bar(x + width/2, full_values, width, label='Full Sensors')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Performance Comparison: IMU Only vs Full Sensors')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(imu_values):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(full_values):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'sensor_comparison.png'))
    
    # Clean up temporary file
    if os.path.exists(imu_data_path):
        os.remove(imu_data_path)
    
    return {
        'full_f1_binary': full_f1_binary,
        'full_f1_macro': full_f1_macro,
        'full_combined': full_combined,
        'imu_f1_binary': imu_f1_binary,
        'imu_f1_macro': imu_f1_macro,
        'imu_combined': imu_combined,
        'improvement': improvement,
        'percent_improvement': percent_improvement
    }


def main():
    """Main function to run the pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Run pipeline based on mode
    if args.mode == 'train':
        # If IMU-only mode is selected, filter data
        if args.imu_only:
            logger.info("Using IMU-only data (no thermopile or ToF)")
            # Load data
            preprocessor = TimeSeriesPreprocessor(
                window_size=args.window_size,
                step_size=args.step_size,
                normalize=args.normalize
            )
            data = preprocessor.load_data(os.path.join(args.data_dir, 'train_series.parquet'))
            data = filter_sensor_columns(data, imu_only=True)
            
            # Save filtered data to temporary file
            temp_data_path = os.path.join(args.output_dir, 'temp_imu_data.parquet')
            data.to_parquet(temp_data_path)
            
            # Update data path
            args.data_dir = args.output_dir
            
            # Run training
            model, model_path = run_training(args)
            
            # Clean up temporary file
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
        else:
            run_training(args)
    elif args.mode == 'predict':
        run_prediction(args)
    elif args.mode == 'full':
        model, model_path = run_training(args)
        run_prediction(args, model_path=model_path)
    elif args.mode == 'compare':
        compare_sensors(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()