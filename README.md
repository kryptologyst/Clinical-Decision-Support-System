# Clinical Decision Support System (CDSS)

A research-ready implementation of a Clinical Decision Support System for chronic disease management using EHR/tabular data with advanced ML models and explainability.

## ⚠️ IMPORTANT DISCLAIMER

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

This Clinical Decision Support System (CDSS) is a research demonstration project and is **NOT INTENDED FOR CLINICAL USE**. 

- **NOT FOR DIAGNOSIS**: This system does not provide medical diagnosis, treatment recommendations, or clinical decisions
- **NOT MEDICAL ADVICE**: Any outputs from this system should not be considered medical advice or replace professional medical judgment
- **RESEARCH ONLY**: This is a research demonstration showcasing AI/ML techniques in healthcare
- **CLINICIAN SUPERVISION REQUIRED**: If used in any clinical context, it must be under direct supervision of qualified healthcare professionals

## Features

### Core Functionality
- **Rule-based CDSS**: Traditional clinical decision rules for hypertension and diabetes management
- **ML-enhanced CDSS**: Advanced machine learning models (XGBoost, LightGBM, TabNet) for risk prediction
- **Comprehensive Risk Assessment**: Multi-factor risk scoring with clinical interpretation
- **Clinical Recommendations**: Evidence-based recommendations with confidence scores and priority levels

### Advanced ML Capabilities
- **Multiple Model Types**: Gradient boosting (XGBoost, LightGBM) and deep tabular models (TabNet)
- **Model Ensembling**: Weighted ensemble of multiple models for improved performance
- **Patient-level Splitting**: Prevents data leakage in clinical data
- **Class Imbalance Handling**: SMOTE, ADASYN, and other balancing techniques
- **Cross-validation**: Stratified k-fold cross-validation for robust evaluation

### Explainability & Safety
- **SHAP Explanations**: Feature importance and individual prediction explanations
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **Uncertainty Quantification**: Monte Carlo dropout and ensemble uncertainty
- **Safety Checks**: Confidence thresholds, OOD detection, and safety flags
- **Calibration Analysis**: Expected Calibration Error (ECE) and reliability diagrams

### Clinical Evaluation
- **Clinically Meaningful Metrics**: Sensitivity, specificity, PPV, NPV, likelihood ratios
- **Calibration Assessment**: Brier score, calibration curves, decision curve analysis
- **Fairness Evaluation**: Performance across demographic groups
- **Model Leaderboard**: Comprehensive comparison of model performance

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/kryptologyst/Clinical-Decision-Support-System.git
cd Clinical-Decision-Support-System

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Optional Dependencies
```bash
# For SHAP explanations
pip install shap

# For LIME explanations
pip install lime

# For advanced tabular models
pip install pytorch-tabnet rtdl

# For serving
pip install fastapi uvicorn
```

## Quick Start

### 1. Run the Demo
```bash
# Start the Streamlit demo
streamlit run demo/app.py
```

### 2. Train Models
```bash
# Train models on synthetic data
python scripts/train_models.py --config configs/default.yaml
```

### 3. Evaluate Performance
```bash
# Run comprehensive evaluation
python scripts/evaluate_models.py --config configs/default.yaml
```

### 4. Generate Explanations
```bash
# Generate SHAP and LIME explanations
python scripts/generate_explanations.py --config configs/default.yaml
```

## Usage

### Basic CDSS Usage
```python
from src.models.cdss import ClinicalDecisionSupportSystem
from src.data.synthetic_data import generate_synthetic_patients
from src.utils.config import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Initialize CDSS
cdss = ClinicalDecisionSupportSystem(config)

# Generate synthetic patients
patients = generate_synthetic_patients(100, config)

# Train ML models
cdss.train_ml_models(patients)

# Get recommendations for a patient
patient = patients[0]
recommendations = cdss.get_recommendations(patient)

# Display recommendations
cdss.display_recommendations(patient, recommendations)
```

### Advanced Model Training
```python
from src.data.data_pipeline import DataPipeline
from src.models.ml_models import train_models
from src.metrics.clinical_metrics import ClinicalMetrics

# Initialize data pipeline
pipeline = DataPipeline(config)

# Process data
processed_data = pipeline.process_pipeline(n_patients=1000)

# Train models
models = train_models(
    processed_data['X_train'], 
    processed_data['y_train_binary'], 
    config
)

# Evaluate models
metrics = ClinicalMetrics(config)
results = []
for model_name, model in models.items():
    y_pred = model.predict(processed_data['X_test'])
    y_proba = model.predict_proba(processed_data['X_test'])
    
    result = metrics.evaluate_model_performance(
        processed_data['y_test_binary'], y_pred, y_proba, model_name
    )
    results.append(result)

# Create leaderboard
leaderboard = metrics.create_leaderboard(results)
print(leaderboard)
```

## Project Structure

```
0478_Clinical_decision_support_system/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── cdss.py              # Main CDSS class
│   │   └── ml_models.py         # ML model implementations
│   ├── data/                    # Data handling
│   │   ├── synthetic_data.py    # Synthetic data generation
│   │   └── data_pipeline.py     # Data preprocessing pipeline
│   ├── metrics/                 # Evaluation metrics
│   │   └── clinical_metrics.py  # Clinical evaluation metrics
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration management
│       └── explainability.py    # Explainability and uncertainty
├── configs/                     # Configuration files
│   └── default.yaml            # Default configuration
├── demo/                        # Demo applications
│   └── app.py                  # Streamlit demo
├── scripts/                     # Training and evaluation scripts
├── tests/                       # Unit tests
├── assets/                      # Generated assets (plots, reports)
├── data/                        # Data directory
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data
├── requirements.txt            # Python dependencies
├── DISCLAIMER.md              # Important disclaimers
└── README.md                  # This file
```

## Configuration

The system uses YAML configuration files. Key configuration sections:

### Data Configuration
```yaml
data:
  n_patients: 1000
  test_size: 0.2
  val_size: 0.2
  random_state: 42
  patient_level_split: true
```

### Model Configuration
```yaml
models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  lightgbm:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
```

### Clinical Rules
```yaml
clinical_rules:
  hypertension:
    systolic_threshold: 140
    diastolic_threshold: 90
  diabetes:
    glucose_threshold: 126
    hba1c_threshold: 6.5
```

## Evaluation Metrics

### Classification Metrics
- **AUC-ROC**: Area under the ROC curve
- **AUC-PRC**: Area under the Precision-Recall curve
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **PPV**: Positive predictive value
- **NPV**: Negative predictive value
- **F1-Score**: Harmonic mean of precision and recall

### Calibration Metrics
- **Brier Score**: Mean squared error of probability predictions
- **ECE**: Expected Calibration Error
- **MCE**: Maximum Calibration Error
- **Calibration Curves**: Reliability diagrams

### Clinical Metrics
- **Likelihood Ratios**: LR+ and LR- for clinical interpretation
- **Decision Curve Analysis**: Net benefit analysis
- **Risk Stratification**: Performance across risk categories

## Explainability

### SHAP Explanations
- **Global Explanations**: Feature importance across the dataset
- **Local Explanations**: Individual prediction explanations
- **Summary Plots**: Feature impact visualization
- **Waterfall Plots**: Step-by-step prediction breakdown

### LIME Explanations
- **Local Interpretability**: Model-agnostic explanations
- **Feature Perturbation**: Understanding model behavior
- **Explanation Stability**: Consistency across similar samples

### Uncertainty Quantification
- **Monte Carlo Dropout**: Bayesian uncertainty estimation
- **Ensemble Uncertainty**: Model disagreement analysis
- **Out-of-Distribution Detection**: Identifying novel cases
- **Prediction Intervals**: Confidence bounds for predictions

## Safety Features

### Data Privacy
- **No PHI Logging**: Protected health information is not logged
- **Synthetic Data**: Demo uses synthetic patient data
- **De-identification**: Built-in utilities for data anonymization

### Model Safety
- **Confidence Thresholds**: Minimum confidence requirements
- **Uncertainty Checks**: High uncertainty warnings
- **OOD Detection**: Out-of-distribution sample identification
- **Safety Flags**: Automated safety assessment

### Clinical Safety
- **Evidence Levels**: Recommendation evidence strength
- **Confidence Scores**: Recommendation confidence levels
- **Actionable Flags**: Clear indication of actionable recommendations
- **Disclaimers**: Prominent research-only warnings

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pre-commit

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/
ruff check src/
```

### Code Style
- **Type Hints**: All functions should have type hints
- **Docstrings**: NumPy/Google style docstrings
- **Formatting**: Black for formatting, Ruff for linting
- **Testing**: Pytest for unit tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cdss_demo,
  title={Clinical Decision Support System for Chronic Disease Management},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Clinical-Decision-Support-System},
  note={Research demonstration - not for clinical use}
}
```

---

**Remember: This system is for research and educational purposes only. It is NOT intended for clinical use.**
# Clinical-Decision-Support-System
