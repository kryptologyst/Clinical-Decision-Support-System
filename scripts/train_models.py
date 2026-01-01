"""
Training script for Clinical Decision Support System models.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.data.data_pipeline import DataPipeline
from src.models.ml_models import train_models
from src.metrics.clinical_metrics import ClinicalMetrics
from src.utils.config import load_config, set_deterministic_seed, get_device
from src.utils.explainability import ExplainabilityEngine


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def train_and_evaluate_models(config_path: str, n_patients: int = 1000):
    """Train and evaluate models.
    
    Args:
        config_path: Path to configuration file
        n_patients: Number of patients to generate
    """
    logger = setup_logging()
    logger.info("Starting model training and evaluation")
    logger.info("DISCLAIMER: This is for research purposes only. Not for clinical use.")
    
    # Load configuration
    config = load_config(config_path)
    set_deterministic_seed(config.get('seed', 42))
    
    # Create output directories
    Path("logs").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Initialize data pipeline
    logger.info("Initializing data pipeline...")
    pipeline = DataPipeline(config)
    
    # Process data
    logger.info(f"Processing data for {n_patients} patients...")
    processed_data = pipeline.process_pipeline(
        n_patients=n_patients,
        balance_data=True,
        balance_method='smote'
    )
    
    # Train models
    logger.info("Training models...")
    models = train_models(
        pd.DataFrame(processed_data['X_train'], columns=processed_data['feature_names']),
        processed_data['y_train_binary'],
        config
    )
    
    logger.info(f"Trained {len(models)} models: {list(models.keys())}")
    
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
        
        # Save model
        model_path = f"checkpoints/{model_name}_model.joblib"
        model.save(model_path)
        logger.info(f"Saved {model_name} model to {model_path}")
    
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
    
    # Setup explainability
    logger.info("Setting up explainability...")
    explainability_engine = ExplainabilityEngine(config)
    
    # Setup SHAP explainers
    explainability_engine.setup_shap_explainers(
        models, 
        processed_data['X_train'], 
        processed_data['feature_names']
    )
    
    # Setup LIME explainer
    explainability_engine.setup_lime_explainer(
        processed_data['X_train'], 
        processed_data['feature_names']
    )
    
    # Generate explanations for test set
    logger.info("Generating explanations...")
    explanations = {}
    
    for model_name in models.keys():
        try:
            shap_data = explainability_engine.get_shap_explanations(
                model_name, processed_data['X_test'][:50]  # Limit to 50 samples
            )
            if shap_data:
                explanations[model_name] = shap_data
                
                # Save SHAP values
                shap_path = f"assets/shap_values_{model_name}_{timestamp}.joblib"
                joblib.dump(shap_data, shap_path)
                logger.info(f"Saved SHAP values for {model_name} to {shap_path}")
                
        except Exception as e:
            logger.warning(f"Failed to generate SHAP explanations for {model_name}: {e}")
    
    # Save explainability engine
    explainability_path = f"checkpoints/explainability_engine_{timestamp}.joblib"
    joblib.dump(explainability_engine, explainability_path)
    logger.info(f"Saved explainability engine to {explainability_path}")
    
    # Print summary
    logger.info("Training and evaluation completed successfully!")
    logger.info(f"Models trained: {list(models.keys())}")
    logger.info(f"Test set size: {len(processed_data['X_test'])}")
    logger.info(f"Best model (by AUC-ROC): {leaderboard.iloc[0]['Model']}")
    logger.info(f"Best AUC-ROC: {leaderboard.iloc[0]['AUC-ROC']:.4f}")
    
    # Print leaderboard
    logger.info("\nModel Leaderboard:")
    logger.info(leaderboard.round(4).to_string(index=False))
    
    return models, results, leaderboard


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train CDSS models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--n_patients", 
        type=int, 
        default=1000,
        help="Number of patients to generate"
    )
    
    args = parser.parse_args()
    
    try:
        models, results, leaderboard = train_and_evaluate_models(
            args.config, args.n_patients
        )
        print("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
