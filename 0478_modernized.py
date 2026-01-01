"""
Training and evaluation pipeline for Clinical Decision Support System.

This script demonstrates the complete pipeline from data generation
to model training and evaluation.
"""

import sys
from pathlib import Path
import logging

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.models.cdss import ClinicalDecisionSupportSystem
from src.data.synthetic_data import generate_synthetic_patients
from src.utils.config import load_config, set_deterministic_seed


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("CLINICAL DECISION SUPPORT SYSTEM - DEMONSTRATION")
    print("=" * 80)
    print("DISCLAIMER: This is for research purposes only. Not for clinical use.")
    print("=" * 80)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config("configs/default.yaml")
    set_deterministic_seed(config.get('seed', 42))
    
    # Initialize CDSS
    logger.info("Initializing Clinical Decision Support System...")
    cdss = ClinicalDecisionSupportSystem(config)
    
    # Generate synthetic patient data for demo
    logger.info("Generating synthetic patient data...")
    patients = generate_synthetic_patients(n_patients=100, config=config)
    
    # Train ML models
    logger.info("Training ML models...")
    cdss.train_ml_models(patients)
    
    # Run demo on sample patients
    logger.info("Running CDSS recommendations on sample patients...")
    
    for i, patient in enumerate(patients[:5]):  # Show first 5 patients
        print(f"\n--- Patient {i+1}: {patient.patient_id} ---")
        print(f"Age: {patient.age}, Gender: {patient.gender}")
        print(f"BP: {patient.systolic_bp:.0f}/{patient.diastolic_bp:.0f} mmHg")
        print(f"Glucose: {patient.fasting_glucose:.0f} mg/dL, HbA1c: {patient.hba1c:.1f}%")
        print(f"BMI: {patient.bmi:.1f}, Cholesterol: {patient.cholesterol:.0f} mg/dL")
        print(f"Risk Score: {patient.risk_score:.2f}")
        
        # Get recommendations
        recommendations = cdss.get_recommendations(patient)
        
        print(f"\nClinical Recommendations:")
        for j, rec in enumerate(recommendations, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.priority, "⚪")
            print(f"{j}. {priority_emoji} [{rec.category}] {rec.recommendation}")
            print(f"   Rationale: {rec.rationale}")
            print(f"   Confidence: {rec.confidence:.1%} | Evidence: {rec.evidence_level}")
    
    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("To run the interactive demo:")
    print("  streamlit run demo/app.py")
    print("\nTo train models:")
    print("  python scripts/train_models.py")
    print("\nTo evaluate models:")
    print("  python scripts/evaluate_models.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
