"""
Clinical Decision Support System (CDSS) for Chronic Disease Management

A modern, research-ready implementation of a CDSS for hypertension and diabetes management
using EHR/tabular data with advanced ML models and explainability.

DISCLAIMER: This is for research and educational purposes only. Not for clinical use.
"""

import os
import sys
import warnings
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the CDSS demo."""
    logger.info("Starting Clinical Decision Support System Demo")
    logger.info("DISCLAIMER: This is for research purposes only. Not for clinical use.")
    
    # Import and run the modernized CDSS
    from src.models.cdss import ClinicalDecisionSupportSystem
    from src.data.synthetic_data import generate_synthetic_patients
    from src.utils.config import load_config
    
    # Load configuration
    config = load_config("configs/default.yaml")
    
    # Initialize CDSS
    cdss = ClinicalDecisionSupportSystem(config)
    
    # Generate synthetic patient data for demo
    patients = generate_synthetic_patients(n_patients=100, config=config)
    
    # Run demo on sample patients
    logger.info("Running CDSS recommendations on sample patients...")
    for i, patient in enumerate(patients[:5]):  # Show first 5 patients
        logger.info(f"\n--- Patient {i+1} ---")
        recommendations = cdss.get_recommendations(patient)
        cdss.display_recommendations(patient, recommendations)

if __name__ == "__main__":
    main()
