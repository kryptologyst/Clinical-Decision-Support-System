"""
Unit tests for Clinical Decision Support System.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.models.cdss import ClinicalDecisionSupportSystem, Recommendation, RiskAssessment
from src.data.synthetic_data import PatientData, SyntheticDataGenerator
from src.data.data_pipeline import DataPipeline
from src.models.ml_models import XGBoostModel, LightGBMModel
from src.metrics.clinical_metrics import ClinicalMetrics
from src.utils.config import load_config, set_deterministic_seed, get_device


class TestPatientData:
    """Test PatientData class."""
    
    def test_patient_data_creation(self):
        """Test PatientData object creation."""
        patient = PatientData(
            patient_id="TEST_001",
            age=50,
            gender="male",
            systolic_bp=140,
            diastolic_bp=90,
            fasting_glucose=126,
            hba1c=6.5,
            bmi=28,
            cholesterol=200,
            hdl_cholesterol=50,
            ldl_cholesterol=120,
            triglycerides=150,
            smoking_status="former",
            family_history_diabetes=True,
            family_history_hypertension=False,
            exercise_frequency="moderate",
            diet_quality="good",
            stress_level="moderate",
            sleep_hours=7.5,
            medication_compliance=0.8,
            has_diabetes=True,
            has_hypertension=True,
            has_cardiovascular_disease=False,
            risk_score=0.6
        )
        
        assert patient.patient_id == "TEST_001"
        assert patient.age == 50
        assert patient.gender == "male"
        assert patient.has_diabetes is True
        assert patient.has_hypertension is True
        assert patient.risk_score == 0.6


class TestSyntheticDataGenerator:
    """Test SyntheticDataGenerator class."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        config = {'data': {'random_state': 42}}
        generator = SyntheticDataGenerator(config)
        assert generator.config == config
        assert generator.random_state == 42
    
    def test_generate_patient(self):
        """Test single patient generation."""
        config = {'data': {'random_state': 42}}
        generator = SyntheticDataGenerator(config)
        
        patient = generator.generate_patient("TEST_001")
        
        assert isinstance(patient, PatientData)
        assert patient.patient_id == "TEST_001"
        assert 18 <= patient.age <= 85
        assert patient.gender in ["male", "female"]
        assert 0 <= patient.risk_score <= 1
    
    def test_generate_patients(self):
        """Test multiple patients generation."""
        config = {'data': {'random_state': 42}}
        generator = SyntheticDataGenerator(config)
        
        patients = generator.generate_patients(10)
        
        assert len(patients) == 10
        assert all(isinstance(p, PatientData) for p in patients)
        assert all(p.patient_id.startswith("PAT_") for p in patients)
    
    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        config = {'data': {'random_state': 42}}
        generator = SyntheticDataGenerator(config)
        
        patients = generator.generate_patients(5)
        df = generator.to_dataframe(patients)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'patient_id' in df.columns
        assert 'risk_score' in df.columns


class TestClinicalDecisionSupportSystem:
    """Test ClinicalDecisionSupportSystem class."""
    
    def test_cdss_initialization(self):
        """Test CDSS initialization."""
        config = {
            'clinical_rules': {
                'hypertension': {'systolic_threshold': 140, 'diastolic_threshold': 90},
                'diabetes': {'glucose_threshold': 126, 'hba1c_threshold': 6.5}
            }
        }
        
        cdss = ClinicalDecisionSupportSystem(config)
        assert cdss.config == config
        assert cdss.clinical_rules == config['clinical_rules']
    
    def test_get_rule_based_recommendations(self):
        """Test rule-based recommendations."""
        config = {
            'clinical_rules': {
                'hypertension': {'systolic_threshold': 140, 'diastolic_threshold': 90},
                'diabetes': {'glucose_threshold': 126, 'hba1c_threshold': 6.5}
            }
        }
        
        cdss = ClinicalDecisionSupportSystem(config)
        
        # High risk patient
        patient = PatientData(
            patient_id="TEST_001",
            age=60,
            gender="male",
            systolic_bp=150,  # Above threshold
            diastolic_bp=95,  # Above threshold
            fasting_glucose=130,  # Above threshold
            hba1c=7.0,  # Above threshold
            bmi=30,
            cholesterol=220,
            hdl_cholesterol=50,
            ldl_cholesterol=120,
            triglycerides=150,
            smoking_status="current",
            family_history_diabetes=True,
            family_history_hypertension=True,
            exercise_frequency="none",
            diet_quality="poor",
            stress_level="high",
            sleep_hours=6,
            medication_compliance=0.5,
            has_diabetes=True,
            has_hypertension=True,
            has_cardiovascular_disease=True,
            risk_score=0.8
        )
        
        recommendations = cdss._get_rule_based_recommendations(patient)
        
        assert len(recommendations) > 0
        assert all(isinstance(r, Recommendation) for r in recommendations)
        
        # Check for high priority recommendations
        high_priority = [r for r in recommendations if r.priority == 'high']
        assert len(high_priority) > 0  # Should have high priority recommendations
    
    def test_assess_risk(self):
        """Test risk assessment."""
        config = {'clinical_rules': {}}
        cdss = ClinicalDecisionSupportSystem(config)
        
        patient = PatientData(
            patient_id="TEST_001",
            age=50,
            gender="male",
            systolic_bp=120,
            diastolic_bp=80,
            fasting_glucose=100,
            hba1c=5.5,
            bmi=25,
            cholesterol=180,
            hdl_cholesterol=60,
            ldl_cholesterol=100,
            triglycerides=120,
            smoking_status="never",
            family_history_diabetes=False,
            family_history_hypertension=False,
            exercise_frequency="moderate",
            diet_quality="good",
            stress_level="low",
            sleep_hours=8,
            medication_compliance=0.9,
            has_diabetes=False,
            has_hypertension=False,
            has_cardiovascular_disease=False,
            risk_score=0.3
        )
        
        risk_assessment = cdss.assess_risk(patient)
        
        assert isinstance(risk_assessment, RiskAssessment)
        assert risk_assessment.overall_risk == patient.risk_score
        assert risk_assessment.risk_category in ['low', 'moderate', 'high', 'very_high']
        assert isinstance(risk_assessment.risk_factors, list)
        assert isinstance(risk_assessment.protective_factors, list)
        assert isinstance(risk_assessment.recommendations, list)


class TestDataPipeline:
    """Test DataPipeline class."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = {
            'data': {
                'test_size': 0.2,
                'val_size': 0.2,
                'random_state': 42,
                'patient_level_split': True
            }
        }
        
        pipeline = DataPipeline(config)
        assert pipeline.config == config
        assert pipeline.test_size == 0.2
        assert pipeline.val_size == 0.2
        assert pipeline.random_state == 42
    
    def test_load_synthetic_data(self):
        """Test synthetic data loading."""
        config = {'data': {'random_state': 42}}
        pipeline = DataPipeline(config)
        
        X, y = pipeline.load_synthetic_data(100)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert len(X) == 100
        assert len(y) == 100
        assert len(X.columns) > 0
    
    def test_preprocess_features(self):
        """Test feature preprocessing."""
        config = {'data': {'random_state': 42}}
        pipeline = DataPipeline(config)
        
        # Create test data with categorical variables
        X = pd.DataFrame({
            'age': [30, 40, 50],
            'gender': ['male', 'female', 'male'],
            'smoking_status': ['never', 'former', 'current']
        })
        
        X_processed = pipeline.preprocess_features(X, fit=True)
        
        assert isinstance(X_processed, np.ndarray)
        assert X_processed.shape[0] == 3
        assert X_processed.shape[1] == 3  # All features should be numerical
    
    def test_split_data(self):
        """Test data splitting."""
        config = {
            'data': {
                'test_size': 0.2,
                'val_size': 0.2,
                'random_state': 42,
                'patient_level_split': False
            }
        }
        
        pipeline = DataPipeline(config)
        
        X = pd.DataFrame({'feature1': range(100), 'feature2': range(100)})
        y = np.random.randint(0, 2, 100)
        
        splits = pipeline.split_data(X, y)
        
        assert 'X_train' in splits
        assert 'X_val' in splits
        assert 'X_test' in splits
        assert 'y_train' in splits
        assert 'y_val' in splits
        assert 'y_test' in splits
        
        # Check split sizes are approximately correct
        assert len(splits['X_test']) == 20  # 20% of 100
        assert len(splits['X_val']) == 20   # 20% of remaining 80
        assert len(splits['X_train']) == 60  # Remaining


class TestMLModels:
    """Test ML model classes."""
    
    def test_xgboost_model_initialization(self):
        """Test XGBoost model initialization."""
        config = {
            'xgboost': {
                'n_estimators': 10,
                'max_depth': 3,
                'random_state': 42
            }
        }
        
        model = XGBoostModel(config)
        assert model.config == config
        assert model.model is not None
        assert not model.is_fitted
    
    def test_lightgbm_model_initialization(self):
        """Test LightGBM model initialization."""
        config = {
            'lightgbm': {
                'n_estimators': 10,
                'max_depth': 3,
                'random_state': 42
            }
        }
        
        model = LightGBMModel(config)
        assert model.config == config
        assert model.model is not None
        assert not model.is_fitted


class TestClinicalMetrics:
    """Test ClinicalMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        config = {
            'evaluation': {
                'calibration_bins': 10,
                'decision_threshold': 0.5
            }
        }
        
        metrics = ClinicalMetrics(config)
        assert metrics.config == config
        assert metrics.calibration_bins == 10
        assert metrics.decision_threshold == 0.5
    
    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation."""
        config = {'evaluation': {'calibration_bins': 10}}
        metrics = ClinicalMetrics(config)
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4])
        
        result = metrics.calculate_classification_metrics(y_true, y_pred, y_proba)
        
        assert isinstance(result, dict)
        assert 'accuracy' in result
        assert 'sensitivity' in result
        assert 'specificity' in result
        assert 'auc_roc' in result
        assert 'brier_score' in result
        
        # Check metric values are reasonable
        assert 0 <= result['accuracy'] <= 1
        assert 0 <= result['sensitivity'] <= 1
        assert 0 <= result['specificity'] <= 1
        assert 0 <= result['auc_roc'] <= 1
    
    def test_calculate_calibration_metrics(self):
        """Test calibration metrics calculation."""
        config = {'evaluation': {'calibration_bins': 5}}
        metrics = ClinicalMetrics(config)
        
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        
        result = metrics.calculate_calibration_metrics(y_true, y_proba)
        
        assert isinstance(result, dict)
        assert 'ece' in result
        assert 'mce' in result
        assert 'fraction_of_positives' in result
        assert 'mean_predicted_value' in result
        
        # Check calibration metrics are reasonable
        assert 0 <= result['ece'] <= 1
        assert 0 <= result['mce'] <= 1


class TestConfig:
    """Test configuration utilities."""
    
    def test_set_deterministic_seed(self):
        """Test deterministic seed setting."""
        set_deterministic_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in ['cuda', 'mps', 'cpu']
    
    def test_get_default_config(self):
        """Test default configuration generation."""
        from src.utils.config import get_default_config
        
        config = get_default_config()
        
        assert isinstance(config, dict)
        assert 'system' in config
        assert 'data' in config
        assert 'models' in config
        assert 'clinical_rules' in config


if __name__ == "__main__":
    pytest.main([__file__])
