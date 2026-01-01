"""
Advanced ML models for Clinical Decision Support System.

Implements gradient boosting baselines and deep tabular models for EHR/tabular data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import joblib
import warnings

# Gradient boosting models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Deep tabular models
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    warnings.warn("TabNet not available. Install with: pip install pytorch-tabnet")

try:
    from rtdl import FTTransformer
    RTDL_AVAILABLE = True
except ImportError:
    RTDL_AVAILABLE = False
    warnings.warn("RTDL not available. Install with: pip install rtdl")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report


class BaseModel(ABC):
    """Abstract base class for all CDSS models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'BaseModel':
        """Fit the model to training data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        pass
    
    def save(self, filepath: str) -> None:
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str) -> 'BaseModel':
        """Load the model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        return self


class XGBoostModel(BaseModel):
    """XGBoost model for CDSS."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        model_config = config.get('xgboost', {})
        self.model = xgb.XGBClassifier(**model_config)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'XGBoostModel':
        """Fit XGBoost model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        # Prepare features
        X_processed = self._prepare_features(X)
        
        # Fit model
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        X_processed = self._prepare_features(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        X_processed = self._prepare_features(X)
        return self.model.predict_proba(X_processed)
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare features for XGBoost.
        
        Args:
            X: Input features
            
        Returns:
            Processed features
        """
        X_processed = X.copy()
        
        # Encode categorical variables
        for col in X_processed.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col])
            else:
                X_processed[col] = self.label_encoders[col].transform(X_processed[col])
        
        return X_processed.values


class LightGBMModel(BaseModel):
    """LightGBM model for CDSS."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LightGBM model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        model_config = config.get('lightgbm', {})
        self.model = lgb.LGBMClassifier(**model_config)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'LightGBMModel':
        """Fit LightGBM model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        X_processed = self._prepare_features(X)
        self.model.fit(X_processed, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_processed = self._prepare_features(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        X_processed = self._prepare_features(X)
        return self.model.predict_proba(X_processed)
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare features for LightGBM."""
        X_processed = X.copy()
        
        for col in X_processed.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col])
            else:
                X_processed[col] = self.label_encoders[col].transform(X_processed[col])
        
        return X_processed.values


class TabNetModel(BaseModel):
    """TabNet model for CDSS."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TabNet model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet not available. Install with: pip install pytorch-tabnet")
        
        model_config = config.get('tabnet', {})
        self.model = TabNetClassifier(**model_config)
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'TabNetModel':
        """Fit TabNet model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        X_processed = self._prepare_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Fit TabNet
        self.model.fit(
            X_train=X_scaled,
            y_train=y,
            eval_set=[(X_scaled, y)],
            eval_metric=['auc'],
            max_epochs=self.config.get('training', {}).get('epochs', 100),
            patience=self.config.get('training', {}).get('patience', 10),
            batch_size=self.config.get('training', {}).get('batch_size', 32),
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_processed = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        X_processed = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict_proba(X_scaled)
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare features for TabNet."""
        X_processed = X.copy()
        
        for col in X_processed.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col])
            else:
                X_processed[col] = self.label_encoders[col].transform(X_processed[col])
        
        return X_processed.values


class ModelEnsemble:
    """Ensemble of multiple models for improved performance."""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        """Initialize ensemble.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using ensemble.
        
        Args:
            X: Features to predict on
            
        Returns:
            Weighted average of model predictions
        """
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions using ensemble.
        
        Args:
            X: Features to predict on
            threshold: Decision threshold
            
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)


def create_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
    """Factory function to create models.
    
    Args:
        model_type: Type of model to create
        config: Model configuration
        
    Returns:
        Initialized model instance
    """
    if model_type.lower() == 'xgboost':
        return XGBoostModel(config)
    elif model_type.lower() == 'lightgbm':
        return LightGBMModel(config)
    elif model_type.lower() == 'tabnet':
        return TabNetModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_models(X: pd.DataFrame, y: np.ndarray, config: Dict[str, Any]) -> Dict[str, BaseModel]:
    """Train multiple models and return them.
    
    Args:
        X: Training features
        y: Training targets
        config: Configuration dictionary
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # Train XGBoost
    try:
        xgb_model = XGBoostModel(config)
        xgb_model.fit(X, y)
        models['xgboost'] = xgb_model
    except Exception as e:
        print(f"Failed to train XGBoost: {e}")
    
    # Train LightGBM
    try:
        lgb_model = LightGBMModel(config)
        lgb_model.fit(X, y)
        models['lightgbm'] = lgb_model
    except Exception as e:
        print(f"Failed to train LightGBM: {e}")
    
    # Train TabNet (if available)
    if TABNET_AVAILABLE:
        try:
            tabnet_model = TabNetModel(config)
            tabnet_model.fit(X, y)
            models['tabnet'] = tabnet_model
        except Exception as e:
            print(f"Failed to train TabNet: {e}")
    
    return models
