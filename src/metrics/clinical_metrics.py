"""
Clinical evaluation metrics and calibration for CDSS.

Implements clinically meaningful metrics including calibration, decision curves,
and fairness evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings


class ClinicalMetrics:
    """Clinical evaluation metrics for CDSS."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize clinical metrics.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_config = config.get('evaluation', {})
        self.calibration_bins = self.metrics_config.get('calibration_bins', 10)
        self.decision_threshold = self.metrics_config.get('decision_threshold', 0.5)
        
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of classification metrics
        """
        metrics = {}
        
        # Basic classification metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = metrics['sensitivity']
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
        
        # Clinical metrics
        metrics['ppv'] = metrics['precision']  # Positive Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # Likelihood ratios
        metrics['lr_positive'] = metrics['sensitivity'] / (1 - metrics['specificity']) if (1 - metrics['specificity']) > 0 else np.inf
        metrics['lr_negative'] = (1 - metrics['sensitivity']) / metrics['specificity'] if metrics['specificity'] > 0 else 0.0
        
        # AUC metrics
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc_roc'] = 0.5
            
        try:
            metrics['auc_prc'] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics['auc_prc'] = 0.0
        
        # Calibration metrics
        metrics['brier_score'] = brier_score_loss(y_true, y_proba)
        metrics['log_loss'] = log_loss(y_true, y_proba)
        
        return metrics
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate calibration metrics.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of calibration metrics
        """
        calibration_metrics = {}
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=self.calibration_bins
        )
        
        calibration_metrics['fraction_of_positives'] = fraction_of_positives
        calibration_metrics['mean_predicted_value'] = mean_predicted_value
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        calibration_metrics['ece'] = ece
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        calibration_metrics['mce'] = mce
        
        return calibration_metrics
    
    def calculate_decision_curve_analysis(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate decision curve analysis metrics.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with decision curve data
        """
        # Threshold probabilities
        thresholds = np.linspace(0, 1, 101)
        
        # Calculate net benefit for each threshold
        net_benefits = []
        treat_all_benefits = []
        treat_none_benefits = []
        
        prevalence = y_true.mean()
        
        for threshold in thresholds:
            # Predictions at this threshold
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            
            # Net benefit calculation
            net_benefit = (tp / len(y_true)) - (fp / len(y_true)) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)
            
            # Treat all benefit
            treat_all_benefit = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
            treat_all_benefits.append(treat_all_benefit)
            
            # Treat none benefit
            treat_none_benefits.append(0)
        
        return {
            'thresholds': thresholds,
            'net_benefits': np.array(net_benefits),
            'treat_all_benefits': np.array(treat_all_benefits),
            'treat_none_benefits': np.array(treat_none_benefits)
        }
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_proba: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        results = {
            'model_name': model_name,
            'classification_metrics': self.calculate_classification_metrics(y_true, y_pred, y_proba),
            'calibration_metrics': self.calculate_calibration_metrics(y_true, y_proba),
            'decision_curve_analysis': self.calculate_decision_curve_analysis(y_true, y_proba)
        }
        
        return results
    
    def create_leaderboard(self, model_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a model leaderboard.
        
        Args:
            model_results: List of model evaluation results
            
        Returns:
            DataFrame with model rankings
        """
        leaderboard_data = []
        
        for result in model_results:
            model_name = result['model_name']
            metrics = result['classification_metrics']
            calibration = result['calibration_metrics']
            
            leaderboard_data.append({
                'Model': model_name,
                'AUC-ROC': metrics['auc_roc'],
                'AUC-PRC': metrics['auc_prc'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'PPV': metrics['ppv'],
                'NPV': metrics['npv'],
                'F1-Score': metrics['f1_score'],
                'Brier Score': metrics['brier_score'],
                'ECE': calibration['ece'],
                'MCE': calibration['mce']
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        
        # Sort by AUC-ROC (primary metric)
        leaderboard = leaderboard.sort_values('AUC-ROC', ascending=False).reset_index(drop=True)
        
        return leaderboard
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                              model_name: str = "Model", ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot calibration curve.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            ax: Matplotlib axes object
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Model calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=self.calibration_bins
        )
        
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                label=f'{model_name} (ECE: {self.calculate_calibration_metrics(y_true, y_proba)["ece"]:.3f})')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_decision_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                           model_name: str = "Model", ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot decision curve analysis.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            ax: Matplotlib axes object
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        dca_data = self.calculate_decision_curve_analysis(y_true, y_proba)
        
        ax.plot(dca_data['thresholds'], dca_data['net_benefits'], 
                label=f'{model_name}', linewidth=2)
        ax.plot(dca_data['thresholds'], dca_data['treat_all_benefits'], 
                '--', label='Treat All', alpha=0.7)
        ax.plot(dca_data['thresholds'], dca_data['treat_none_benefits'], 
                '--', label='Treat None', alpha=0.7)
        
        ax.set_xlabel('Threshold Probability')
        ax.set_ylabel('Net Benefit')
        ax.set_title(f'Decision Curve Analysis - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      model_name: str = "Model", ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot ROC curve.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            ax: Matplotlib axes object
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                   model_name: str = "Model", ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot precision-recall curve.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            ax: Matplotlib axes object
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc_prc = average_precision_score(y_true, y_proba)
        
        ax.plot(recall, precision, label=f'{model_name} (AUC-PRC = {auc_prc:.3f})', linewidth=2)
        
        # Baseline (random classifier)
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def evaluate_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray, sensitive_attributes: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate model fairness across sensitive attributes.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities
            sensitive_attributes: Dictionary of sensitive attribute arrays
            
        Returns:
            Dictionary with fairness evaluation results
        """
        fairness_results = {}
        
        for attr_name, attr_values in sensitive_attributes.items():
            unique_values = np.unique(attr_values)
            attr_results = {}
            
            for value in unique_values:
                mask = attr_values == value
                if mask.sum() > 0:  # Ensure we have samples for this group
                    group_y_true = y_true[mask]
                    group_y_pred = y_pred[mask]
                    group_y_proba = y_proba[mask]
                    
                    group_metrics = self.calculate_classification_metrics(
                        group_y_true, group_y_pred, group_y_proba
                    )
                    attr_results[f'{attr_name}_{value}'] = group_metrics
            
            fairness_results[attr_name] = attr_results
        
        return fairness_results
    
    def generate_evaluation_report(self, model_results: List[Dict[str, Any]], 
                                 save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            model_results: List of model evaluation results
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("CLINICAL DECISION SUPPORT SYSTEM - EVALUATION REPORT")
        report.append("=" * 80)
        report.append("DISCLAIMER: This evaluation is for research purposes only.")
        report.append("Not intended for clinical use.")
        report.append("")
        
        # Leaderboard
        leaderboard = self.create_leaderboard(model_results)
        report.append("MODEL LEADERBOARD")
        report.append("-" * 40)
        report.append(leaderboard.round(4).to_string(index=False))
        report.append("")
        
        # Detailed results for each model
        for result in model_results:
            model_name = result['model_name']
            metrics = result['classification_metrics']
            calibration = result['calibration_metrics']
            
            report.append(f"DETAILED RESULTS - {model_name}")
            report.append("-" * 40)
            report.append(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            report.append(f"AUC-PRC: {metrics['auc_prc']:.4f}")
            report.append(f"Sensitivity: {metrics['sensitivity']:.4f}")
            report.append(f"Specificity: {metrics['specificity']:.4f}")
            report.append(f"PPV: {metrics['ppv']:.4f}")
            report.append(f"NPV: {metrics['npv']:.4f}")
            report.append(f"F1-Score: {metrics['f1_score']:.4f}")
            report.append(f"Brier Score: {metrics['brier_score']:.4f}")
            report.append(f"ECE: {calibration['ece']:.4f}")
            report.append(f"MCE: {calibration['mce']:.4f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
