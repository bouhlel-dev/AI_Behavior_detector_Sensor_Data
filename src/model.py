import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BFRBModel:
    """Class for training and evaluating models for BFRB detection."""
    
    def __init__(self, model_type='rf', class_weights=None, random_state=42):
        """
        Initialize the model.
        
        Args:
            model_type (str): Type of model to use ('rf', 'xgb', 'gb', 'lr')
            class_weights (dict, optional): Class weights for handling imbalance
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type
        self.class_weights = class_weights
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        logger.info(f"Initialized {model_type} model")
        
    def _create_model(self):
        """
        Create the model based on model_type.
        
        Returns:
            object: Initialized model
        """
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=300,  # Increased for better performance
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight=self.class_weights,
                random_state=self.random_state,
                n_jobs=-1,
                bootstrap=True,
                max_features='sqrt',  # Better for high-dimensional data
                criterion='gini'  # Default, but explicitly stated
            )
        elif self.model_type == 'xgb':
            return XGBClassifier(
                n_estimators=300,  # Increased for better performance
                max_depth=6,
                learning_rate=0.03,  # Reduced for better generalization
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,  # Added for better feature selection
                min_child_weight=3,  # Added to prevent overfitting
                gamma=0.1,  # Minimum loss reduction for split
                random_state=self.random_state,
                n_jobs=-1,
                scale_pos_weight=1.0,  # Will be adjusted based on class distribution
                tree_method='hist',  # Faster algorithm
                eval_metric='mlogloss'  # Multiclass log loss
            )
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=300,  # Increased for better performance
                max_depth=4,  # Increased from 3
                learning_rate=0.03,  # Reduced for better generalization
                subsample=0.8,  # Added for better generalization
                min_samples_split=5,  # Added to prevent overfitting
                min_samples_leaf=2,  # Added to prevent overfitting
                random_state=self.random_state,
                max_features='sqrt',  # Better for high-dimensional data
                validation_fraction=0.1  # For early stopping
            )
        elif self.model_type == 'lr':
            return LogisticRegression(
                C=0.3,  # Reduced for stronger regularization
                penalty='l2',
                solver='saga',  # Better for large datasets
                class_weight=self.class_weights,
                random_state=self.random_state,
                max_iter=3000,  # Increased for convergence
                n_jobs=-1,
                multi_class='multinomial'  # Better for multiclass problems
            )
        elif self.model_type == 'stacked':
            # Create a stacked ensemble of models
            from sklearn.ensemble import StackingClassifier
            
            # Base estimators
            estimators = [
                ('rf', RandomForestClassifier(
                    n_estimators=200, 
                    random_state=self.random_state,
                    class_weight=self.class_weights,
                    n_jobs=-1
                )),
                ('xgb', XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=self.random_state,
                    n_jobs=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=2
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=self.random_state,
                    subsample=0.8,
                    max_depth=4
                ))
            ]
            
            # Final estimator
            final_estimator = LogisticRegression(
                C=0.5,
                class_weight=self.class_weights,
                random_state=self.random_state,
                max_iter=2000,
                multi_class='multinomial'
            )
            
            # Create stacked model
            return StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=5,  # 5-fold cross-validation for meta-learner training
                stack_method='predict_proba',  # Use predicted probabilities as features
                n_jobs=-1,
                passthrough=False  # Don't include original features
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y, series_ids=None, validation_split=0.2, cv=None):
        """
        Train the model on the given data.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            series_ids (list, optional): Series IDs for group-based CV
            validation_split (float): Validation split ratio
            cv (int, optional): Number of CV folds
            
        Returns:
            object: Trained model
        """
        logger.info(f"Training {self.model_type} model")
        
        # Create model
        self.model = self._create_model()
        
        # If CV is specified, perform cross-validation
        if cv is not None:
            logger.info(f"Performing {cv}-fold cross-validation")
            
            # Choose CV strategy based on whether series_ids are provided
            if series_ids is not None:
                # Use GroupKFold to ensure series_ids don't leak between train/val
                cv_strategy = GroupKFold(n_splits=cv)
                splits = cv_strategy.split(X, y, groups=series_ids)
            else:
                # Use StratifiedKFold to maintain class distribution
                cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                splits = cv_strategy.split(X, y)
            
            # Perform CV
            cv_scores = []
            for i, (train_idx, val_idx) in enumerate(splits):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale features
                X_train = self.scaler.fit_transform(X_train)
                X_val = self.scaler.transform(X_val)
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = self.model.predict(X_val)
                score = f1_score(y_val, y_pred, average='macro')
                cv_scores.append(score)
                logger.info(f"Fold {i+1}/{cv} - Macro F1: {score:.4f}")
            
            logger.info(f"CV Macro F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            
            # Retrain on full dataset
            logger.info("Retraining on full dataset")
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
        else:
            # Simple train/validation split
            if series_ids is not None:
                # Split by series_id to prevent leakage
                unique_series = np.unique(series_ids)
                train_series, val_series = train_test_split(
                    unique_series, test_size=validation_split, random_state=self.random_state
                )
                
                train_mask = np.isin(series_ids, train_series)
                val_mask = np.isin(series_ids, val_series)
                
                X_train, X_val = X[train_mask], X[val_mask]
                y_train, y_val = y[train_mask], y[val_mask]
            else:
                # Regular stratified split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, stratify=y, random_state=self.random_state
                )
            
            # Scale features
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            score = f1_score(y_val, y_pred, average='macro')
            logger.info(f"Validation Macro F1: {score:.4f}")
            
            # Print classification report
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_val, y_pred))
            
            # Plot confusion matrix
            self._plot_confusion_matrix(y_val, y_pred)
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    def save_model(self, path):
        """
        Save model to disk.
        
        Args:
            path (str): Path to save model
        """
        import joblib
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model from disk.
        
        Args:
            path (str): Path to load model from
        """
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        logger.info(f"Model loaded from {path}")
        
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from the model.
        
        Args:
            feature_names (list, optional): Names of features
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return None
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
        
        return importance_df