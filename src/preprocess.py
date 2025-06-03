import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Class for preprocessing time series data for BFRB detection."""
    
    def __init__(self, window_size=1000, step_size=500, normalize=True):
        """
        Initialize the preprocessor with window parameters.
        
        Args:
            window_size (int): Size of the window in milliseconds
            step_size (int): Step size for sliding window in milliseconds
            normalize (bool): Whether to normalize the data
        """
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        logger.info(f"Initialized preprocessor with window_size={window_size}ms, step_size={step_size}ms")
    
    def load_data(self, file_path):
        """
        Load time series data from parquet file.
        
        Args:
            file_path (str): Path to the parquet file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading data from {file_path}")
        try:
            data = pd.read_parquet(file_path)
            logger.info(f"Loaded data with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_labels(self, file_path):
        """
        Load labels from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded labels
        """
        logger.info(f"Loading labels from {file_path}")
        try:
            labels = pd.read_csv(file_path)
            logger.info(f"Loaded labels with shape {labels.shape}")
            return labels
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            raise
    
    def normalize_data(self, data):
        """
        Normalize sensor data using z-score normalization.
        
        Args:
            data (pd.DataFrame): Data to normalize
            
        Returns:
            pd.DataFrame: Normalized data
        """
        logger.info("Normalizing data")
        # Get sensor columns (excluding series_id and timestamp)
        sensor_cols = [col for col in data.columns if col not in ['series_id', 'timestamp']]
        
        # Group by series_id and normalize each group separately
        normalized_data = data.copy()
        for series_id in tqdm(data['series_id'].unique(), desc="Normalizing series"):
            series_mask = data['series_id'] == series_id
            series_data = data.loc[series_mask, sensor_cols]
            
            # Apply z-score normalization
            mean = series_data.mean()
            std = series_data.std()
            normalized_data.loc[series_mask, sensor_cols] = (series_data - mean) / (std + 1e-8)
        
        logger.info("Data normalization complete")
        return normalized_data
    
    def create_windows(self, data, labels=None):
        """
        Create fixed-size windows from time series data.
        
        Args:
            data (pd.DataFrame): Time series data
            labels (pd.DataFrame, optional): Labels for the data
            
        Returns:
            tuple: (X, y) where X is the windowed features and y is the corresponding labels
        """
        logger.info(f"Creating windows with size={self.window_size}ms, step={self.step_size}ms")
        
        windows = []
        window_labels = []
        window_series_ids = []
        
        # Process each series separately
        for series_id in tqdm(data['series_id'].unique(), desc="Creating windows"):
            series_data = data[data['series_id'] == series_id].sort_values('timestamp')
            
            # Get sensor columns
            sensor_cols = [col for col in series_data.columns if col not in ['series_id', 'timestamp']]
            
            # Get timestamps
            timestamps = series_data['timestamp'].values
            
            # Get start and end times
            start_time = timestamps[0]
            end_time = timestamps[-1]
            
            # Create windows
            window_start = start_time
            while window_start + self.window_size <= end_time:
                window_end = window_start + self.window_size
                
                # Get data in window
                window_mask = (series_data['timestamp'] >= window_start) & (series_data['timestamp'] < window_end)
                window_data = series_data.loc[window_mask, sensor_cols].values
                
                # Skip windows with insufficient data
                if len(window_data) < 10:  # Minimum number of points in a window
                    window_start += self.step_size
                    continue
                
                windows.append(window_data)
                window_series_ids.append(series_id)
                
                # If labels are provided, get the label for this window
                if labels is not None:
                    series_label = labels[labels['series_id'] == series_id]['label'].values[0]
                    window_labels.append(series_label)
                
                window_start += self.step_size
        
        logger.info(f"Created {len(windows)} windows")
        
        if labels is not None:
            return windows, window_labels, window_series_ids
        else:
            return windows, window_series_ids
    
    def extract_features(self, windows):
        """
        Extract features from windows.
        
        Args:
            windows (list): List of windowed data
            
        Returns:
            np.ndarray: Extracted features
        """
        logger.info("Extracting features from windows")
        
        features = []
        for window in tqdm(windows, desc="Extracting features"):
            window_features = []
            
            # Get sensor columns
            imu_cols = [i for i, col in enumerate(window.shape[1]) if 'acc_' in col or 'rot_' in col]
            tof_cols = [i for i, col in enumerate(window.shape[1]) if 'tof_' in col]
            thm_cols = [i for i, col in enumerate(window.shape[1]) if 'thm_' in col]
            
            # Extract IMU features (accelerometer and gyroscope)
            if imu_cols:
                imu_data = window[:, imu_cols]
                
                # Separate accelerometer and gyroscope data if both are present
                acc_cols = [i for i, col in enumerate(imu_cols) if 'acc_' in col]
                rot_cols = [i for i, col in enumerate(imu_cols) if 'rot_' in col]
                
                # Basic statistical features
                window_features.extend([
                    np.mean(imu_data, axis=0),  # Mean
                    np.std(imu_data, axis=0),   # Standard deviation
                    np.min(imu_data, axis=0),   # Minimum
                    np.max(imu_data, axis=0),   # Maximum
                    np.median(imu_data, axis=0),  # Median
                    np.percentile(imu_data, 25, axis=0),  # 25th percentile
                    np.percentile(imu_data, 75, axis=0),  # 75th percentile
                    np.percentile(imu_data, 10, axis=0),  # 10th percentile
                    np.percentile(imu_data, 90, axis=0),  # 90th percentile
                ])
                
                # Calculate jerk (derivative of acceleration)
                if imu_data.shape[0] > 1:
                    jerk = np.diff(imu_data, axis=0)
                    window_features.extend([
                        np.mean(np.abs(jerk), axis=0),  # Mean absolute jerk
                        np.std(jerk, axis=0),          # Std of jerk
                        np.max(np.abs(jerk), axis=0),  # Max absolute jerk
                    ])
                    
                    # Second derivative (jounce/snap)
                    if jerk.shape[0] > 1:
                        jounce = np.diff(jerk, axis=0)
                        window_features.extend([
                            np.mean(np.abs(jounce), axis=0),  # Mean absolute jounce
                            np.max(np.abs(jounce), axis=0),  # Max absolute jounce
                        ])
                
                # Frequency domain features (FFT)
                fft_features = np.abs(np.fft.fft(imu_data, axis=0))
                # Use only first half of FFT (up to Nyquist frequency)
                fft_features = fft_features[1:fft_features.shape[0]//2 + 1]
                window_features.extend([
                    np.mean(fft_features, axis=0),  # Mean of FFT
                    np.std(fft_features, axis=0),   # Std of FFT
                    np.max(fft_features, axis=0),   # Max of FFT
                ])
                
                # Calculate dominant frequency and its magnitude
                dominant_freq_idx = np.argmax(fft_features, axis=0)
                dominant_freq_mag = np.max(fft_features, axis=0)
                window_features.extend([
                    dominant_freq_idx,      # Dominant frequency index
                    dominant_freq_mag,      # Magnitude of dominant frequency
                ])
                
                # Calculate energy in different frequency bands
                if fft_features.shape[0] > 10:
                    # Low frequency band (0-20% of Nyquist)
                    low_band = fft_features[:int(0.2 * fft_features.shape[0])]
                    # Mid frequency band (20-60% of Nyquist)
                    mid_band = fft_features[int(0.2 * fft_features.shape[0]):int(0.6 * fft_features.shape[0])]
                    # High frequency band (60-100% of Nyquist)
                    high_band = fft_features[int(0.6 * fft_features.shape[0]):]
                    
                    window_features.extend([
                        np.sum(low_band, axis=0),   # Energy in low frequency band
                        np.sum(mid_band, axis=0),   # Energy in mid frequency band
                        np.sum(high_band, axis=0),  # Energy in high frequency band
                    ])
                
                # Calculate cross-correlation between axes if multiple axes exist
                if imu_data.shape[1] > 1:
                    for i in range(imu_data.shape[1]):
                        for j in range(i+1, imu_data.shape[1]):
                            # Calculate correlation coefficient
                            corr = np.corrcoef(imu_data[:, i], imu_data[:, j])[0, 1]
                            window_features.append(corr)
            
            # Extract ToF features if available
            if tof_cols:
                tof_data = window[:, tof_cols]
                # Check if we have valid ToF data (not all -1)
                if not np.all(tof_data == -1):
                    # Reshape ToF data to expected format for feature extraction
                    # Assuming 5 ToF sensors with 64 pixels each
                    tof_reshaped = tof_data.reshape(tof_data.shape[0], 5, 64)
                    from utils import extract_tof_features
                    tof_features = extract_tof_features(tof_reshaped)
                    if len(tof_features) > 0:
                        window_features.append(tof_features)
            
            # Extract thermopile features if available
            if thm_cols:
                thm_data = window[:, thm_cols]
                # Check if we have valid thermopile data (not all NaN)
                if not np.all(np.isnan(thm_data)):
                    from utils import extract_thermopile_features
                    thm_features = extract_thermopile_features(thm_data)
                    if len(thm_features) > 0:
                        window_features.append(thm_features)
            
            # Flatten and append
            features.append(np.concatenate([f.flatten() for f in window_features if f.size > 0]))
        
        features = np.array(features)
        logger.info(f"Extracted features with shape {features.shape}")
        return features
    
    def preprocess(self, data_path, labels_path=None):
        """Preprocess data and extract features."""
        # Load data
        data = self.load_data(data_path)
        
        # Handle missing sensor data
        data = self.handle_missing_data(data)
        
        # Normalize data if needed
        if self.normalize:
            data = self.normalize_data(data)
        
        # Load labels if provided
        labels = None
        if labels_path is not None:
            labels = self.load_labels(labels_path)
        
        # Create windows
        if labels is not None:
            windows, window_labels, window_series_ids = self.create_windows(data, labels)
        else:
            windows, window_series_ids = self.create_windows(data)
        
        # Extract features
        features = self.extract_features(windows)
        
        if labels is not None:
            return features, np.array(window_labels), window_series_ids
        else:
            return features, window_series_ids
            
    def handle_missing_data(self, data):
        """
        Handle missing sensor data for both ToF and thermopile sensors.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        logger.info("Handling missing sensor data")
        
        # Handle ToF sensor data
        tof_cols = [col for col in data.columns if 'tof_' in col]
        if tof_cols:
            from utils import handle_missing_sensor_data
            data = handle_missing_sensor_data(data, sensor_type='tof')
            logger.info("Handled missing ToF sensor data")
        
        # Handle thermopile sensor data
        thm_cols = [col for col in data.columns if 'thm_' in col]
        if thm_cols:
            from utils import handle_missing_sensor_data
            data = handle_missing_sensor_data(data, sensor_type='thm')
            logger.info("Handled missing thermopile sensor data")
        
        return data