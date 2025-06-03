import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list, optional): Names of classes
    """
    if class_names is None:
        class_names = ['No BFRB', 'Hair Pulling', 'Nail Biting', 'Skin Picking']
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calculate F1 score (macro)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    logger.info(f"Macro F1 Score: {macro_f1:.4f}")
    
    return macro_f1


def plot_feature_importance(importance, feature_names, top_n=20):
    """
    Plot feature importance.
    
    Args:
        importance (np.ndarray): Feature importance values
        feature_names (list): Names of features
        top_n (int): Number of top features to show
    """
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Take top N features
    if top_n is not None and len(importance_df) > top_n:
        importance_df = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return importance_df


def plot_sensor_data(data, series_id, label=None, n_samples=1000):
    """
    Plot sensor data for a specific series_id.
    
    Args:
        data (pd.DataFrame): Sensor data
        series_id (str): Series ID to plot
        label (int, optional): Label for the series
        n_samples (int): Maximum number of samples to plot
    """
    # Get data for the series_id
    series_data = data[data['series_id'] == series_id].sort_values('timestamp')
    
    # Get label name if provided
    label_name = ''
    if label is not None:
        label_names = ['No BFRB', 'Hair Pulling', 'Nail Biting', 'Skin Picking']
        label_name = f" (Label: {label_names[label]})"
    
    # Sample data if too large
    if len(series_data) > n_samples:
        series_data = series_data.sample(n_samples, random_state=42)
        series_data = series_data.sort_values('timestamp')
    
    # Get sensor columns
    sensor_cols = [col for col in series_data.columns if col not in ['series_id', 'timestamp']]
    acc_cols = [col for col in sensor_cols if 'acc' in col]
    gyr_cols = [col for col in sensor_cols if 'gyr' in col]
    other_cols = [col for col in sensor_cols if col not in acc_cols and col not in gyr_cols]
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot accelerometer data
    for col in acc_cols:
        axes[0].plot(series_data['timestamp'], series_data[col], label=col)
    axes[0].set_title(f'Accelerometer Data for Series {series_id}{label_name}')
    axes[0].set_ylabel('Acceleration')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot gyroscope data
    for col in gyr_cols:
        axes[1].plot(series_data['timestamp'], series_data[col], label=col)
    axes[1].set_title(f'Gyroscope Data for Series {series_id}{label_name}')
    axes[1].set_ylabel('Angular Velocity')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot other data if available
    if other_cols:
        for col in other_cols:
            axes[2].plot(series_data['timestamp'], series_data[col], label=col)
        axes[2].set_title(f'Other Sensor Data for Series {series_id}{label_name}')
        axes[2].set_ylabel('Value')
        axes[2].legend()
        axes[2].grid(True)
    
    # Set common x-axis label
    axes[-1].set_xlabel('Timestamp')
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels):
    """
    Plot class distribution.
    
    Args:
        labels (pd.DataFrame): DataFrame containing labels
    """
    # Count labels
    label_counts = labels['label'].value_counts().sort_index()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=label_counts.index, y=label_counts.values)
    
    # Add count labels on top of bars
    for i, count in enumerate(label_counts.values):
        ax.text(i, count + 5, str(count), ha='center')
    
    plt.title('Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(range(len(label_counts)), ['No BFRB', 'Hair Pulling', 'Nail Biting', 'Skin Picking'])
    plt.tight_layout()
    plt.show()
    
    # Print class distribution statistics
    total = label_counts.sum()
    for i, count in enumerate(label_counts.values):
        percentage = 100 * count / total
        logger.info(f"Class {i}: {count} samples ({percentage:.2f}%)")
    
    # Calculate class imbalance ratio (majority / minority)
    imbalance_ratio = label_counts.max() / label_counts.min()
    logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    return label_counts


def extract_tof_features(tof_data):
    """
    Extract features from time-of-flight sensor data.
    
    Args:
        tof_data (np.ndarray): Time-of-flight sensor data with shape (n_samples, n_sensors, 64)
                              where 64 is the number of pixels (8x8 grid)
    
    Returns:
        np.ndarray: Extracted features
    """
    if tof_data is None or len(tof_data) == 0:
        return np.array([])
    
    features = []
    
    # Process each sensor separately
    for sensor_idx in range(tof_data.shape[1]):
        sensor_data = tof_data[:, sensor_idx, :]
        
        # Skip if all values are -1 (no sensor response)
        if np.all(sensor_data == -1):
            continue
        
        # Replace -1 with NaN for proper statistics calculation
        sensor_data = np.where(sensor_data == -1, np.nan, sensor_data)
        
        # Basic statistical features (ignoring NaN values)
        features.extend([
            np.nanmean(sensor_data, axis=0),  # Mean per pixel
            np.nanstd(sensor_data, axis=0),   # Std per pixel
            np.nanmin(sensor_data, axis=0),   # Min per pixel
            np.nanmax(sensor_data, axis=0),   # Max per pixel
            np.nanmedian(sensor_data, axis=0),  # Median per pixel
            np.nanpercentile(sensor_data, 25, axis=0),  # 25th percentile per pixel
            np.nanpercentile(sensor_data, 75, axis=0),  # 75th percentile per pixel
        ])
        
        # Calculate spatial gradient features (approximation of object movement)
        # Reshape to 8x8 grid for spatial analysis
        grid_data = sensor_data.reshape(sensor_data.shape[0], 8, 8)
        
        # Calculate horizontal and vertical gradients
        h_gradient = np.diff(grid_data, axis=1)
        v_gradient = np.diff(grid_data, axis=2)
        
        # Extract statistics from gradients
        features.extend([
            np.nanmean(np.abs(h_gradient), axis=0).flatten(),  # Mean horizontal gradient
            np.nanstd(np.abs(h_gradient), axis=0).flatten(),  # Std horizontal gradient
            np.nanmean(np.abs(v_gradient), axis=0).flatten(),  # Mean vertical gradient
            np.nanstd(np.abs(v_gradient), axis=0).flatten(),  # Std vertical gradient
        ])
        
        # Count valid pixels over time (object presence)
        valid_pixel_count = np.sum(~np.isnan(sensor_data), axis=1)
        features.extend([
            np.mean(valid_pixel_count),  # Average number of valid pixels
            np.std(valid_pixel_count),   # Std of valid pixel count
            np.max(valid_pixel_count),   # Max valid pixels (closest approach)
            np.min(valid_pixel_count),   # Min valid pixels
        ])
        
        # Temporal features - changes over time
        if sensor_data.shape[0] > 1:
            # Calculate temporal gradients (changes in proximity over time)
            temp_gradient = np.diff(sensor_data, axis=0)
            features.extend([
                np.nanmean(np.abs(temp_gradient), axis=0),  # Mean temporal change
                np.nanstd(np.abs(temp_gradient), axis=0),   # Std of temporal change
                np.nanmax(np.abs(temp_gradient), axis=0),    # Max temporal change
            ])
            
            # Calculate frequency domain features for repetitive patterns
            if sensor_data.shape[0] > 10:  # Need enough samples for FFT
                # Fill NaN values for FFT
                filled_data = np.nan_to_num(sensor_data, nan=0)
                # Calculate FFT along time axis
                fft_features = np.abs(np.fft.fft(filled_data, axis=0))
                # Use only first few components
                fft_features = fft_features[1:min(10, fft_features.shape[0]), :]
                features.extend([
                    np.mean(fft_features, axis=0),  # Mean of frequency components
                    np.std(fft_features, axis=0),   # Std of frequency components
                    np.max(fft_features, axis=0),    # Dominant frequency component
                ])
    
    # Flatten and concatenate all features
    return np.concatenate([f.flatten() for f in features if f.size > 0])


def extract_thermopile_features(thm_data):
    """
    Extract features from thermopile sensor data.
    
    Args:
        thm_data (np.ndarray): Thermopile sensor data with shape (n_samples, n_sensors)
    
    Returns:
        np.ndarray: Extracted features
    """
    if thm_data is None or len(thm_data) == 0:
        return np.array([])
    
    features = []
    
    # Basic statistical features
    features.extend([
        np.nanmean(thm_data, axis=0),  # Mean per sensor
        np.nanstd(thm_data, axis=0),   # Std per sensor
        np.nanmin(thm_data, axis=0),   # Min per sensor
        np.nanmax(thm_data, axis=0),   # Max per sensor
        np.nanmedian(thm_data, axis=0),  # Median per sensor
        np.nanpercentile(thm_data, 25, axis=0),  # 25th percentile
        np.nanpercentile(thm_data, 75, axis=0),  # 75th percentile
        np.nanpercentile(thm_data, 90, axis=0),  # 90th percentile - captures high temp events
    ])
    
    # Temperature gradient features (changes over time)
    if thm_data.shape[0] > 1:
        temp_gradient = np.diff(thm_data, axis=0)
        features.extend([
            np.nanmean(np.abs(temp_gradient), axis=0),  # Mean absolute gradient
            np.nanstd(np.abs(temp_gradient), axis=0),   # Std of gradient
            np.nanmax(np.abs(temp_gradient), axis=0),   # Max absolute gradient
            np.nanmedian(np.abs(temp_gradient), axis=0),  # Median absolute gradient
        ])
        
        # Second derivative (acceleration of temperature change)
        if temp_gradient.shape[0] > 1:
            temp_accel = np.diff(temp_gradient, axis=0)
            features.extend([
                np.nanmean(np.abs(temp_accel), axis=0),  # Mean absolute acceleration
                np.nanmax(np.abs(temp_accel), axis=0),   # Max absolute acceleration
            ])
    
    # Sensor correlation features
    if thm_data.shape[1] > 1:  # Only if we have multiple sensors
        # Calculate correlation between sensors
        corr_matrix = np.corrcoef(thm_data.T)
        # Extract upper triangle of correlation matrix (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
        features.append(upper_tri)  # Correlation between sensors
        
        # Calculate pairwise differences between sensors
        for i in range(thm_data.shape[1]):
            for j in range(i+1, thm_data.shape[1]):
                sensor_diff = thm_data[:, i] - thm_data[:, j]
                features.extend([
                    np.nanmean(sensor_diff),  # Mean difference
                    np.nanstd(sensor_diff),   # Std of difference
                    np.nanmax(sensor_diff),   # Max difference
                    np.nanmin(sensor_diff),   # Min difference
                ])
    
    # Frequency domain features for repetitive patterns
    if thm_data.shape[0] > 10:  # Need enough samples for FFT
        # Fill NaN values for FFT
        filled_data = np.nan_to_num(thm_data, nan=0)
        # Calculate FFT along time axis
        fft_features = np.abs(np.fft.fft(filled_data, axis=0))
        # Use only first few components
        fft_features = fft_features[1:min(10, fft_features.shape[0]), :]
        features.extend([
            np.mean(fft_features, axis=0),  # Mean of frequency components
            np.std(fft_features, axis=0),   # Std of frequency components
            np.max(fft_features, axis=0),    # Dominant frequency component
        ])
    
    # Duration of elevated temperature (potential skin contact)
    if thm_data.shape[0] > 1:
        # Calculate threshold as mean + 1 std for each sensor
        thresholds = np.nanmean(thm_data, axis=0) + np.nanstd(thm_data, axis=0)
        # Count consecutive frames above threshold
        for i in range(thm_data.shape[1]):
            above_threshold = thm_data[:, i] > thresholds[i]
            # Count runs of True values
            runs = np.diff(np.concatenate(([0], above_threshold.astype(int), [0])))
            runs_starts = np.where(runs == 1)[0]
            runs_ends = np.where(runs == -1)[0]
            if len(runs_starts) > 0 and len(runs_ends) > 0:
                run_lengths = runs_ends - runs_starts
                features.extend([
                    np.mean(run_lengths),  # Average duration of elevated temp
                    np.max(run_lengths),   # Max duration of elevated temp
                    len(run_lengths),      # Number of elevated temp events
                ])
            else:
                # No runs found, add zeros
                features.extend([0, 0, 0])
    
    # Flatten and concatenate all features
    return np.concatenate([f.flatten() for f in features if f.size > 0])


def handle_missing_sensor_data(data, sensor_type='tof'):
    """
    Handle missing sensor data by imputing or creating indicator features.
    
    Args:
        data (pd.DataFrame): Input data
        sensor_type (str): Type of sensor ('tof' or 'thm')
    
    Returns:
        pd.DataFrame: Data with handled missing values
    """
    logger.info(f"Handling missing {sensor_type} sensor data")
    
    # Create a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Get sensor columns
    if sensor_type == 'tof':
        sensor_cols = [col for col in data.columns if 'tof_' in col]
    elif sensor_type == 'thm':
        sensor_cols = [col for col in data.columns if 'thm_' in col]
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    # Check if all sensor data is missing for each sequence
    sequence_ids = data['series_id'].unique()
    for seq_id in sequence_ids:
        seq_mask = data['series_id'] == seq_id
        seq_data = data.loc[seq_mask, sensor_cols]
        
        # If all values are null, this is an IMU-only sequence
        if seq_data.isnull().all().all():
            # For IMU-only sequences, fill with -1 (special value)
            processed_data.loc[seq_mask, sensor_cols] = -1
            logger.debug(f"Sequence {seq_id} is IMU-only, filled {sensor_type} with -1")
        else:
            # For sequences with partial missing data, use forward fill then backward fill
            # This maintains the temporal nature of the data
            seq_filled = seq_data.fillna(method='ffill').fillna(method='bfill')
            processed_data.loc[seq_mask, sensor_cols] = seq_filled
            logger.debug(f"Filled partial missing {sensor_type} data in sequence {seq_id}")
    
    # Create indicator features for IMU-only sequences
    processed_data[f'{sensor_type}_available'] = ~(processed_data[sensor_cols].iloc[:, 0] == -1).astype(int)
    
    return processed_data