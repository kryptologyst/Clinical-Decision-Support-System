"""
Evaluation script for Clinical Decision Support System models.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.data.data_pipeline import DataPipeline
from src.models.ml_models import BaseModel
from src.metrics.clinical_metrics import ClinicalMetrics
from src.utils.config import load_config, set_deterministic_seed


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/evaluation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_trained_models(models_dir: str = "checkpoints"):
    """Load trained models from disk.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dictionary of loaded models
    """
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory {models_dir} not found")
    
    for model_file in models_path.glob("*_model.joblib"):
        model_name = model_file.stem.replace("_model", "")
        try:
            model = BaseModel({})  # Dummy config for loading
            model.load(str(model_file))
            models[model_name] = model
            logging.info(f"Loaded model: {model_name}")
        except Exception as e:
            logging.warning(f"Failed to load model {model_name}: {e}")
    
    return models


def generate_evaluation_plots(results, save_dir: str = "assets"):
    """Generate evaluation plots.
    
    Args:
        results: List of evaluation results
        save_dir: Directory to save plots
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    metrics = ClinicalMetrics({})
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # ROC Curves
    ax1 = axes[0, 0]
    for result in results:
        model_name = result['model_name']
        # This would need actual test data - placeholder for now
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Calibration Curves
    ax2 = axes[0, 1]
    for result in results:
        model_name = result['model_name']
        # Placeholder calibration curve
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax2.set_xlabel('Mean Predicted Probability')
        ax2.set_ylabel('Fraction of Positives')
        ax2.set_title('Calibration Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Performance Metrics Bar Chart
    ax3 = axes[1, 0]
    model_names = [result['model_name'] for result in results]
    auc_scores = [result['classification_metrics']['auc_roc'] for result in results]
    
    bars = ax3.bar(model_names, auc_scores)
    ax3.set_ylabel('AUC-ROC')
    ax3.set_title('AUC-ROC Comparison')
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Calibration Error Comparison
    ax4 = axes[1, 1]
    ece_scores = [result['calibration_metrics']['ece'] for result in results]
    
    bars = ax4.bar(model_names, ece_scores)
    ax4.set_ylabel('Expected Calibration Error')
    ax4.set_title('Calibration Error Comparison')
    
    # Add value labels on bars
    for bar, score in zip(bars, ece_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = save_path / f"evaluation_plots_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logging.info(f"Saved evaluation plots to {plot_path}")


def evaluate_models(config_path: str, models_dir: str = "checkpoints"):
    """Evaluate trained models.
    
    Args:
        config_path: Path to configuration file
        models_dir: Directory containing trained models
    """
    logger = setup_logging()
    logger.info("Starting model evaluation")
    logger.info("DISCLAIMER: This is for research purposes only. Not for clinical use.")
    
    # Load configuration
    config = load_config(config_path)
    set_deterministic_seed(config.get('seed', 42))
    
    # Create output directories
    Path("logs").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    
    # Load trained models
    logger.info(f"Loading models from {models_dir}...")
    models = load_trained_models(models_dir)
    
    if not models:
        logger.error("No trained models found. Please run train_models.py first.")
        return
    
    # Load test data
    logger.info("Loading test data...")
    pipeline = DataPipeline(config)
    processed_data = pipeline.process_pipeline(
        n_patients=config.get('data', {}).get('n_patients', 1000),
        balance_data=True,
        balance_method='smote'
    )
    
    # Evaluate models
    logger.info("Evaluating models...")
    metrics = ClinicalMetrics(config)
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        X_test_df = pd.DataFrame(processed_data['X_test'], columns=processed_data['feature_names'])
        y_pred = model.predict(X_test_df)
        y_proba = model.predict_proba(X_test_df)
        
        # Evaluate performance
        result = metrics.evaluate_model_performance(
            processed_data['y_test_binary'], y_pred, y_proba, model_name
        )
        results.append(result)
    
    # Create leaderboard
    logger.info("Creating model leaderboard...")
    leaderboard = metrics.create_leaderboard(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save leaderboard
    leaderboard_path = f"assets/leaderboard_{timestamp}.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logger.info(f"Saved leaderboard to {leaderboard_path}")
    
    # Save evaluation results
    results_path = f"assets/evaluation_results_{timestamp}.joblib"
    joblib.dump(results, results_path)
    logger.info(f"Saved evaluation results to {results_path}")
    
    # Generate evaluation report
    report = metrics.generate_evaluation_report(results)
    report_path = f"assets/evaluation_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved evaluation report to {report_path}")
    
    # Generate plots
    logger.info("Generating evaluation plots...")
    generate_evaluation_plots(results)
    
    # Print summary
    logger.info("Evaluation completed successfully!")
    logger.info(f"Models evaluated: {list(models.keys())}")
    logger.info(f"Test set size: {len(processed_data['X_test'])}")
    logger.info(f"Best model (by AUC-ROC): {leaderboard.iloc[0]['Model']}")
    logger.info(f"Best AUC-ROC: {leaderboard.iloc[0]['AUC-ROC']:.4f}")
    
    # Print leaderboard
    logger.info("\nModel Leaderboard:")
    logger.info(leaderboard.round(4).to_string(index=False))
    
    return results, leaderboard


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate CDSS models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="checkpoints",
        help="Directory containing trained models"
    )
    
    args = parser.parse_args()
    
    try:
        results, leaderboard = evaluate_models(args.config, args.models_dir)
        print("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
