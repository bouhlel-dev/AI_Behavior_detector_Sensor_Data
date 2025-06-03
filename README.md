# CMI - Detect Behavior with Sensor Data

This repository contains code for the [CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data) Kaggle competition. The goal is to detect body-focused repetitive behaviors (BFRBs) from wrist-worn sensor data.

## Project Structure

```
cmi-bfrb-detection/
│
├── data/                      # Downloaded Kaggle data
│   ├── train_series.parquet
│   ├── train_labels.csv
│   ├── test_series.parquet
│   └── sample_submission.csv
│
├── notebooks/                 # EDA and experiments
│   └── 01_eda.ipynb
│
├── src/
│   ├── preprocess.py          # Feature extraction & windowing
│   ├── train.py               # Model training
│   ├── model.py               # Model architecture
│   └── predict.py             # Inference & submission creation
│
├── outputs/                   # Model weights, logs, submission files
│
├── requirements.txt
└── README.md
```

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the competition data from Kaggle and place it in the `data/` directory

## Usage

### Training

To train a model with default parameters:

```bash
python src/train.py --data_dir data --output_dir outputs
```

Options:
- `--window_size`: Size of the window in milliseconds (default: 1000)
- `--step_size`: Step size for sliding window in milliseconds (default: 500)
- `--normalize`: Whether to normalize the data (default: True)
- `--model_type`: Type of model to use ('rf', 'xgb', 'gb', 'lr') (default: 'rf')
- `--random_state`: Random seed for reproducibility (default: 42)
- `--cv`: Number of cross-validation folds (0 for no CV) (default: 5)
- `--class_weight`: Whether to use class weights (default: True)

### Prediction

To generate predictions for the test set:

```bash
python src/predict.py --data_dir data --output_dir outputs --model_path outputs/rf_model.pkl
```

Options:
- `--window_size`: Size of the window in milliseconds (default: 1000)
- `--step_size`: Step size for sliding window in milliseconds (default: 500)
- `--normalize`: Whether to normalize the data (default: True)
- `--model_path`: Path to the trained model (required)

## Methodology

### Preprocessing

- Normalize sensor values using z-score normalization
- Segment time-series into fixed-size windows
- Extract statistical and frequency domain features from each window

### Modeling

- Classical ML models: Random Forest, XGBoost, Gradient Boosting, Logistic Regression
- Address class imbalance using class weights
- Cross-validation with GroupKFold based on series_id

### Evaluation

- Macro-averaged F1 score across all classes
- Confusion matrix visualization
- Feature importance analysis

## Target Labels

| Label | Description                      |
|-------|----------------------------------|
| 0     | No BFRB                          |
| 1     | Hair Pulling                     |
| 2     | Nail Biting                      |
| 3     | Skin Picking                     |

