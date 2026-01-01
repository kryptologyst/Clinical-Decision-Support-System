"""
Explainability and uncertainty quantification for CDSS.

Implements SHAP explanations, uncertainty quantification, and safety checks
for clinical decision support.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import logging

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class ExplainabilityEngine:
    """Explainability engine for CDSS models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize explainability engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # SHAP explainers
        self.shap_explainers = {}
        self.shap_values = {}
        
        # LIME explainer
        self.lime_explainer = None
        
        # Feature names
        self.feature_names = None
        
    def setup_shap_explainers(self, models: Dict[str, Any], X_train: np.ndarray, 
                              feature_names: List[str]) -> None:
        """Setup SHAP explainers for different model types.
        
        Args:
            models: Dictionary of trained models
            X_train: Training data
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available, skipping SHAP explainers")
            return
        
        self.feature_names = feature_names
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # For tree-based models, use TreeExplainer
                    if hasattr(model.model, 'get_booster'):  # XGBoost
                        explainer = shap.TreeExplainer(model.model)
                    elif hasattr(model.model, 'booster_'):  # LightGBM
                        explainer = shap.TreeExplainer(model.model)
                    else:
                        # For other models, use KernelExplainer
                        explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
                    
                    self.shap_explainers[model_name] = explainer
                    self.logger.info(f"SHAP explainer created for {model_name}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to create SHAP explainer for {model_name}: {e}")
    
    def setup_lime_explainer(self, X_train: np.ndarray, feature_names: List[str]) -> None:
        """Setup LIME explainer.
        
        Args:
            X_train: Training data
            feature_names: List of feature names
        """
        if not LIME_AVAILABLE:
            self.logger.warning("LIME not available, skipping LIME explainer")
            return
        
        try:
            self.lime_explainer = lime.tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                mode='classification',
                discretize_continuous=True
            )
            self.logger.info("LIME explainer created successfully")
        except Exception as e:
            self.logger.warning(f"Failed to create LIME explainer: {e}")
    
    def get_shap_explanations(self, model_name: str, X: np.ndarray, 
                             max_samples: int = 100) -> Dict[str, Any]:
        """Get SHAP explanations for a model.
        
        Args:
            model_name: Name of the model
            X: Data to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary with SHAP explanations
        """
        if model_name not in self.shap_explainers:
            raise ValueError(f"No SHAP explainer found for {model_name}")
        
        explainer = self.shap_explainers[model_name]
        
        # Limit samples for computational efficiency
        if len(X) > max_samples:
            X_sample = X[:max_samples]
        else:
            X_sample = X
        
        try:
            if isinstance(explainer, shap.TreeExplainer):
                shap_values = explainer.shap_values(X_sample)
            else:
                shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            self.shap_values[model_name] = shap_values
            
            return {
                'shap_values': shap_values,
                'base_value': explainer.expected_value,
                'feature_names': self.feature_names,
                'data': X_sample
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compute SHAP values for {model_name}: {e}")
            return None
    
    def get_lime_explanations(self, model, X: np.ndarray, 
                            max_samples: int = 10) -> List[Dict[str, Any]]:
        """Get LIME explanations for a model.
        
        Args:
            model: Trained model with predict_proba method
            X: Data to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            List of LIME explanations
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized")
        
        explanations = []
        
        # Limit samples for computational efficiency
        n_samples = min(len(X), max_samples)
        
        for i in range(n_samples):
            try:
                explanation = self.lime_explainer.explain_instance(
                    X[i], 
                    model.predict_proba,
                    num_features=len(self.feature_names)
                )
                
                explanations.append({
                    'sample_idx': i,
                    'explanation': explanation,
                    'feature_names': self.feature_names
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to explain sample {i}: {e}")
        
        return explanations
    
    def plot_shap_summary(self, model_name: str, shap_data: Dict[str, Any], 
                         max_features: int = 20, save_path: Optional[str] = None) -> None:
        """Plot SHAP summary.
        
        Args:
            model_name: Name of the model
            shap_data: SHAP explanation data
            max_features: Maximum number of features to show
            save_path: Optional path to save the plot
        """
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available for plotting")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_data['shap_values'],
                shap_data['data'],
                feature_names=shap_data['feature_names'],
                max_display=max_features,
                show=False
            )
            plt.title(f'SHAP Summary Plot - {model_name}')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to create SHAP summary plot: {e}")
    
    def plot_shap_waterfall(self, model_name: str, shap_data: Dict[str, Any], 
                           sample_idx: int = 0, save_path: Optional[str] = None) -> None:
        """Plot SHAP waterfall for a single prediction.
        
        Args:
            model_name: Name of the model
            shap_data: SHAP explanation data
            sample_idx: Index of sample to explain
            save_path: Optional path to save the plot
        """
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available for plotting")
            return
        
        try:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap_data['shap_values'][sample_idx],
                max_display=15,
                show=False
            )
            plt.title(f'SHAP Waterfall Plot - {model_name} (Sample {sample_idx})')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to create SHAP waterfall plot: {e}")
    
    def get_feature_importance(self, model_name: str, shap_data: Dict[str, Any]) -> pd.DataFrame:
        """Get feature importance from SHAP values.
        
        Args:
            model_name: Name of the model
            shap_data: SHAP explanation data
            
        Returns:
            DataFrame with feature importance
        """
        if shap_data is None:
            return pd.DataFrame()
        
        shap_values = shap_data['shap_values']
        feature_names = shap_data['feature_names']
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap_values
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_prediction(self, model, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """Get comprehensive explanation for a single prediction.
        
        Args:
            model: Trained model
            X: Data to explain
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with comprehensive explanation
        """
        explanation = {
            'sample_idx': sample_idx,
            'prediction': model.predict(X[sample_idx:sample_idx+1])[0],
            'probability': model.predict_proba(X[sample_idx:sample_idx+1])[0],
            'feature_values': dict(zip(self.feature_names, X[sample_idx])),
            'shap_explanation': None,
            'lime_explanation': None
        }
        
        # Get SHAP explanation
        try:
            shap_data = self.get_shap_explanations(list(self.shap_explainers.keys())[0], X[sample_idx:sample_idx+1])
            if shap_data:
                explanation['shap_explanation'] = {
                    'shap_values': shap_data['shap_values'][0],
                    'base_value': shap_data['base_value']
                }
        except Exception as e:
            self.logger.warning(f"Failed to get SHAP explanation: {e}")
        
        # Get LIME explanation
        try:
            lime_explanations = self.get_lime_explanations(model, X[sample_idx:sample_idx+1], max_samples=1)
            if lime_explanations:
                explanation['lime_explanation'] = lime_explanations[0]
        except Exception as e:
            self.logger.warning(f"Failed to get LIME explanation: {e}")
        
        return explanation


class UncertaintyQuantification:
    """Uncertainty quantification for CDSS models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize uncertainty quantification.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def monte_carlo_dropout(self, model: nn.Module, X: np.ndarray, 
                          n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Monte Carlo Dropout for uncertainty quantification.
        
        Args:
            model: PyTorch model with dropout layers
            X: Input data
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        model.eval()  # Set to evaluation mode
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Enable dropout during inference
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                
                # Convert to tensor if needed
                if not isinstance(X, torch.Tensor):
                    X_tensor = torch.FloatTensor(X)
                else:
                    X_tensor = X
                
                pred = model(X_tensor)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def ensemble_uncertainty(self, models: List[Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate uncertainty using model ensemble.
        
        Args:
            models: List of trained models
            X: Input data
            
        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        predictions = []
        
        for model in models:
            try:
                pred = model.predict_proba(X)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"Failed to get prediction from model: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions from ensemble models")
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def temperature_scaling(self, logits: np.ndarray, y_true: np.ndarray, 
                           validation_logits: np.ndarray, validation_y_true: np.ndarray) -> float:
        """Temperature scaling for calibration.
        
        Args:
            logits: Model logits
            y_true: True labels
            validation_logits: Validation logits
            validation_y_true: Validation labels
            
        Returns:
            Optimal temperature
        """
        from sklearn.linear_model import LogisticRegression
        
        # Convert logits to probabilities
        probs = 1 / (1 + np.exp(-logits))
        val_probs = 1 / (1 + np.exp(-validation_logits))
        
        # Fit temperature scaling
        lr = LogisticRegression()
        lr.fit(np.log(probs / (1 - probs)).reshape(-1, 1), y_true)
        
        temperature = 1 / lr.coef_[0][0]
        
        return temperature
    
    def calculate_prediction_intervals(self, predictions: np.ndarray, 
                                     confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals.
        
        Args:
            predictions: Array of predictions
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        
        return lower_bound, upper_bound
    
    def detect_out_of_distribution(self, X: np.ndarray, X_train: np.ndarray, 
                                 method: str = 'isolation_forest') -> np.ndarray:
        """Detect out-of-distribution samples.
        
        Args:
            X: Data to check
            X_train: Training data for reference
            method: Method to use ('isolation_forest', 'one_class_svm')
            
        Returns:
            Array of OOD scores
        """
        if method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            detector = IsolationForest(contamination=0.1, random_state=42)
        elif method == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            detector = OneClassSVM(nu=0.1)
        else:
            raise ValueError(f"Unknown OOD detection method: {method}")
        
        detector.fit(X_train)
        ood_scores = detector.decision_function(X)
        
        return ood_scores


class SafetyChecks:
    """Safety checks for CDSS predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize safety checks.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Safety thresholds
        self.confidence_threshold = 0.8
        self.uncertainty_threshold = 0.3
        self.ood_threshold = -0.5
        
    def check_prediction_safety(self, prediction: float, uncertainty: float, 
                               confidence: float, ood_score: float) -> Dict[str, Any]:
        """Check if prediction is safe to use.
        
        Args:
            prediction: Model prediction
            uncertainty: Prediction uncertainty
            confidence: Prediction confidence
            ood_score: Out-of-distribution score
            
        Returns:
            Dictionary with safety assessment
        """
        safety_flags = []
        
        # Check confidence
        if confidence < self.confidence_threshold:
            safety_flags.append(f"Low confidence: {confidence:.2f} < {self.confidence_threshold}")
        
        # Check uncertainty
        if uncertainty > self.uncertainty_threshold:
            safety_flags.append(f"High uncertainty: {uncertainty:.2f} > {self.uncertainty_threshold}")
        
        # Check OOD
        if ood_score < self.ood_threshold:
            safety_flags.append(f"Out-of-distribution: {ood_score:.2f} < {self.ood_threshold}")
        
        # Overall safety assessment
        is_safe = len(safety_flags) == 0
        
        return {
            'is_safe': is_safe,
            'safety_flags': safety_flags,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'ood_score': ood_score,
            'recommendation': 'Use with caution' if not is_safe else 'Safe to use'
        }
    
    def generate_safety_report(self, safety_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive safety report.
        
        Args:
            safety_assessments: List of safety assessments
            
        Returns:
            Dictionary with safety report
        """
        total_predictions = len(safety_assessments)
        safe_predictions = sum(1 for assessment in safety_assessments if assessment['is_safe'])
        
        safety_rate = safe_predictions / total_predictions if total_predictions > 0 else 0
        
        # Count safety flags
        flag_counts = {}
        for assessment in safety_assessments:
            for flag in assessment['safety_flags']:
                flag_type = flag.split(':')[0]
                flag_counts[flag_type] = flag_counts.get(flag_type, 0) + 1
        
        return {
            'total_predictions': total_predictions,
            'safe_predictions': safe_predictions,
            'unsafe_predictions': total_predictions - safe_predictions,
            'safety_rate': safety_rate,
            'flag_counts': flag_counts,
            'overall_assessment': 'Safe' if safety_rate >= 0.9 else 'Needs attention'
        }
