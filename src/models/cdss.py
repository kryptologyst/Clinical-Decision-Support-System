"""
Clinical Decision Support System with rule-based and ML components.

Combines traditional clinical rules with modern ML models for comprehensive
decision support in chronic disease management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from .ml_models import BaseModel, ModelEnsemble, train_models
from ..data.synthetic_data import PatientData
from ..utils.config import get_device


@dataclass
class Recommendation:
    """Clinical recommendation data structure."""
    category: str
    priority: str  # 'high', 'medium', 'low'
    recommendation: str
    rationale: str
    confidence: float
    evidence_level: str  # 'strong', 'moderate', 'weak'
    actionable: bool


@dataclass
class RiskAssessment:
    """Risk assessment data structure."""
    overall_risk: float
    risk_category: str  # 'low', 'moderate', 'high', 'very_high'
    risk_factors: List[str]
    protective_factors: List[str]
    recommendations: List[Recommendation]


class ClinicalDecisionSupportSystem:
    """Main CDSS class combining rules and ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the CDSS.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = get_device()
        
        # Clinical rules configuration
        self.clinical_rules = config.get('clinical_rules', {})
        
        # ML models (will be trained when needed)
        self.ml_models = {}
        self.ensemble_model = None
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'moderate': 0.5,
            'high': 0.7,
            'very_high': 0.9
        }
        
        self.logger.info(f"CDSS initialized with device: {self.device}")
        self.logger.info("DISCLAIMER: This system is for research purposes only. Not for clinical use.")
    
    def train_ml_models(self, patients: List[PatientData], target_column: str = 'risk_score') -> None:
        """Train ML models on patient data.
        
        Args:
            patients: List of patient data
            target_column: Column to use as target variable
        """
        self.logger.info("Training ML models...")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'patient_id': p.patient_id,
            'age': p.age,
            'gender': p.gender,
            'systolic_bp': p.systolic_bp,
            'diastolic_bp': p.diastolic_bp,
            'fasting_glucose': p.fasting_glucose,
            'hba1c': p.hba1c,
            'bmi': p.bmi,
            'cholesterol': p.cholesterol,
            'hdl_cholesterol': p.hdl_cholesterol,
            'ldl_cholesterol': p.ldl_cholesterol,
            'triglycerides': p.triglycerides,
            'smoking_status': p.smoking_status,
            'family_history_diabetes': p.family_history_diabetes,
            'family_history_hypertension': p.family_history_hypertension,
            'exercise_frequency': p.exercise_frequency,
            'diet_quality': p.diet_quality,
            'stress_level': p.stress_level,
            'sleep_hours': p.sleep_hours,
            'medication_compliance': p.medication_compliance,
            'has_diabetes': p.has_diabetes,
            'has_hypertension': p.has_hypertension,
            'has_cardiovascular_disease': p.has_cardiovascular_disease,
            'risk_score': p.risk_score
        } for p in patients])
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col not in ['patient_id', target_column]]
        X = df[feature_columns]
        
        # Create binary target for high risk
        y = (df[target_column] >= 0.5).astype(int)
        
        # Train models
        self.ml_models = train_models(X, y, self.config)
        
        # Create ensemble
        if len(self.ml_models) > 1:
            models_list = list(self.ml_models.values())
            self.ensemble_model = ModelEnsemble(models_list)
        
        self.logger.info(f"Trained {len(self.ml_models)} ML models")
    
    def get_recommendations(self, patient: PatientData) -> List[Recommendation]:
        """Get clinical recommendations for a patient.
        
        Args:
            patient: Patient data
            
        Returns:
            List of clinical recommendations
        """
        recommendations = []
        
        # Rule-based recommendations
        rule_recs = self._get_rule_based_recommendations(patient)
        recommendations.extend(rule_recs)
        
        # ML-based recommendations
        if self.ml_models:
            ml_recs = self._get_ml_based_recommendations(patient)
            recommendations.extend(ml_recs)
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 0), reverse=True)
        
        return recommendations
    
    def _get_rule_based_recommendations(self, patient: PatientData) -> List[Recommendation]:
        """Get rule-based clinical recommendations.
        
        Args:
            patient: Patient data
            
        Returns:
            List of rule-based recommendations
        """
        recommendations = []
        
        # Hypertension management
        htn_rules = self.clinical_rules.get('hypertension', {})
        systolic_threshold = htn_rules.get('systolic_threshold', 140)
        diastolic_threshold = htn_rules.get('diastolic_threshold', 90)
        
        if patient.systolic_bp >= systolic_threshold or patient.diastolic_bp >= diastolic_threshold:
            recommendations.append(Recommendation(
                category='Hypertension',
                priority='high',
                recommendation='Initiate antihypertensive therapy and lifestyle modifications',
                rationale=f'Blood pressure elevated: {patient.systolic_bp}/{patient.diastolic_bp} mmHg',
                confidence=0.9,
                evidence_level='strong',
                actionable=True
            ))
        else:
            recommendations.append(Recommendation(
                category='Hypertension',
                priority='low',
                recommendation='Continue current blood pressure management',
                rationale=f'Blood pressure controlled: {patient.systolic_bp}/{patient.diastolic_bp} mmHg',
                confidence=0.8,
                evidence_level='moderate',
                actionable=False
            ))
        
        # Diabetes management
        diabetes_rules = self.clinical_rules.get('diabetes', {})
        glucose_threshold = diabetes_rules.get('glucose_threshold', 126)
        hba1c_threshold = diabetes_rules.get('hba1c_threshold', 6.5)
        
        if patient.fasting_glucose >= glucose_threshold or patient.hba1c >= hba1c_threshold:
            recommendations.append(Recommendation(
                category='Diabetes',
                priority='high',
                recommendation='Optimize diabetes management and consider medication adjustment',
                rationale=f'Glucose/HbA1c elevated: {patient.fasting_glucose} mg/dL, {patient.hba1c}%',
                confidence=0.9,
                evidence_level='strong',
                actionable=True
            ))
        
        # Cardiovascular risk
        cv_rules = self.clinical_rules.get('cardiovascular', {})
        cholesterol_threshold = cv_rules.get('cholesterol_threshold', 200)
        bmi_threshold = cv_rules.get('bmi_threshold', 30)
        
        if patient.cholesterol >= cholesterol_threshold or patient.bmi >= bmi_threshold:
            recommendations.append(Recommendation(
                category='Cardiovascular',
                priority='medium',
                recommendation='Address cardiovascular risk factors',
                rationale=f'Elevated cholesterol: {patient.cholesterol} mg/dL, BMI: {patient.bmi:.1f}',
                confidence=0.8,
                evidence_level='moderate',
                actionable=True
            ))
        
        # Preventive care
        preventive_rules = self.clinical_rules.get('preventive_care', {})
        colonoscopy_age = preventive_rules.get('colonoscopy_age', 50)
        
        if patient.age >= colonoscopy_age:
            recommendations.append(Recommendation(
                category='Preventive Care',
                priority='medium',
                recommendation='Schedule age-appropriate screening tests',
                rationale=f'Patient age {patient.age} requires preventive screening',
                confidence=0.7,
                evidence_level='moderate',
                actionable=True
            ))
        
        # Lifestyle recommendations
        if patient.smoking_status == 'current':
            recommendations.append(Recommendation(
                category='Lifestyle',
                priority='high',
                recommendation='Smoking cessation counseling and support',
                rationale='Current smoker - major cardiovascular risk factor',
                confidence=0.95,
                evidence_level='strong',
                actionable=True
            ))
        
        if patient.exercise_frequency == 'none':
            recommendations.append(Recommendation(
                category='Lifestyle',
                priority='medium',
                recommendation='Encourage regular physical activity',
                rationale='No current exercise routine',
                confidence=0.8,
                evidence_level='moderate',
                actionable=True
            ))
        
        return recommendations
    
    def _get_ml_based_recommendations(self, patient: PatientData) -> List[Recommendation]:
        """Get ML-based clinical recommendations.
        
        Args:
            patient: Patient data
            
        Returns:
            List of ML-based recommendations
        """
        recommendations = []
        
        # Convert patient to DataFrame for ML prediction
        patient_df = pd.DataFrame([{
            'age': patient.age,
            'gender': patient.gender,
            'systolic_bp': patient.systolic_bp,
            'diastolic_bp': patient.diastolic_bp,
            'fasting_glucose': patient.fasting_glucose,
            'hba1c': patient.hba1c,
            'bmi': patient.bmi,
            'cholesterol': patient.cholesterol,
            'hdl_cholesterol': patient.hdl_cholesterol,
            'ldl_cholesterol': patient.ldl_cholesterol,
            'triglycerides': patient.triglycerides,
            'smoking_status': patient.smoking_status,
            'family_history_diabetes': patient.family_history_diabetes,
            'family_history_hypertension': patient.family_history_hypertension,
            'exercise_frequency': patient.exercise_frequency,
            'diet_quality': patient.diet_quality,
            'stress_level': patient.stress_level,
            'sleep_hours': patient.sleep_hours,
            'medication_compliance': patient.medication_compliance,
            'has_diabetes': patient.has_diabetes,
            'has_hypertension': patient.has_hypertension,
            'has_cardiovascular_disease': patient.has_cardiovascular_disease
        }])
        
        # Get ML prediction
        if self.ensemble_model:
            risk_prob = self.ensemble_model.predict_proba(patient_df)[0][1]
        elif self.ml_models:
            # Use first available model
            model = list(self.ml_models.values())[0]
            risk_prob = model.predict_proba(patient_df)[0][1]
        else:
            risk_prob = patient.risk_score
        
        # Generate ML-based recommendations
        if risk_prob >= 0.7:
            recommendations.append(Recommendation(
                category='ML Risk Assessment',
                priority='high',
                recommendation='High-risk patient requiring immediate attention',
                rationale=f'ML model predicts {risk_prob:.2%} probability of high risk',
                confidence=risk_prob,
                evidence_level='moderate',
                actionable=True
            ))
        elif risk_prob >= 0.5:
            recommendations.append(Recommendation(
                category='ML Risk Assessment',
                priority='medium',
                recommendation='Moderate-risk patient requiring monitoring',
                rationale=f'ML model predicts {risk_prob:.2%} probability of high risk',
                confidence=risk_prob,
                evidence_level='moderate',
                actionable=True
            ))
        
        return recommendations
    
    def assess_risk(self, patient: PatientData) -> RiskAssessment:
        """Comprehensive risk assessment for a patient.
        
        Args:
            patient: Patient data
            
        Returns:
            Risk assessment with recommendations
        """
        # Calculate overall risk
        overall_risk = patient.risk_score
        
        # Determine risk category
        if overall_risk >= self.risk_thresholds['very_high']:
            risk_category = 'very_high'
        elif overall_risk >= self.risk_thresholds['high']:
            risk_category = 'high'
        elif overall_risk >= self.risk_thresholds['moderate']:
            risk_category = 'moderate'
        else:
            risk_category = 'low'
        
        # Identify risk factors
        risk_factors = []
        if patient.systolic_bp >= 140 or patient.diastolic_bp >= 90:
            risk_factors.append('Hypertension')
        if patient.fasting_glucose >= 126 or patient.hba1c >= 6.5:
            risk_factors.append('Diabetes')
        if patient.cholesterol >= 200:
            risk_factors.append('Hyperlipidemia')
        if patient.bmi >= 30:
            risk_factors.append('Obesity')
        if patient.smoking_status == 'current':
            risk_factors.append('Smoking')
        if patient.age >= 65:
            risk_factors.append('Advanced age')
        
        # Identify protective factors
        protective_factors = []
        if patient.exercise_frequency in ['moderate', 'intense']:
            protective_factors.append('Regular exercise')
        if patient.diet_quality in ['good', 'excellent']:
            protective_factors.append('Healthy diet')
        if patient.medication_compliance >= 0.8:
            protective_factors.append('Good medication adherence')
        if patient.sleep_hours >= 7:
            protective_factors.append('Adequate sleep')
        
        # Get recommendations
        recommendations = self.get_recommendations(patient)
        
        return RiskAssessment(
            overall_risk=overall_risk,
            risk_category=risk_category,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            recommendations=recommendations
        )
    
    def display_recommendations(self, patient: PatientData, recommendations: List[Recommendation]) -> None:
        """Display recommendations in a formatted way.
        
        Args:
            patient: Patient data
            recommendations: List of recommendations
        """
        print(f"\n=== Clinical Decision Support for Patient {patient.patient_id} ===")
        print(f"Age: {patient.age}, Gender: {patient.gender}")
        print(f"BP: {patient.systolic_bp:.0f}/{patient.diastolic_bp:.0f} mmHg")
        print(f"Glucose: {patient.fasting_glucose:.0f} mg/dL, HbA1c: {patient.hba1c:.1f}%")
        print(f"BMI: {patient.bmi:.1f}, Cholesterol: {patient.cholesterol:.0f} mg/dL")
        print(f"Risk Score: {patient.risk_score:.2f}")
        
        print(f"\n=== Clinical Recommendations ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec.category}] {rec.recommendation}")
            print(f"   Priority: {rec.priority.upper()}")
            print(f"   Rationale: {rec.rationale}")
            print(f"   Confidence: {rec.confidence:.1%}")
            print(f"   Evidence Level: {rec.evidence_level}")
            print(f"   Actionable: {'Yes' if rec.actionable else 'No'}")
        
        print(f"\nDISCLAIMER: These recommendations are for research purposes only.")
        print(f"Not intended for clinical use. Consult healthcare professionals for medical advice.")
    
    def get_patient_summary(self, patient: PatientData) -> Dict[str, Any]:
        """Get a comprehensive patient summary.
        
        Args:
            patient: Patient data
            
        Returns:
            Dictionary with patient summary
        """
        risk_assessment = self.assess_risk(patient)
        
        return {
            'patient_id': patient.patient_id,
            'demographics': {
                'age': patient.age,
                'gender': patient.gender
            },
            'vital_signs': {
                'systolic_bp': patient.systolic_bp,
                'diastolic_bp': patient.diastolic_bp,
                'bmi': patient.bmi
            },
            'laboratory_values': {
                'fasting_glucose': patient.fasting_glucose,
                'hba1c': patient.hba1c,
                'cholesterol': patient.cholesterol,
                'hdl_cholesterol': patient.hdl_cholesterol,
                'ldl_cholesterol': patient.ldl_cholesterol,
                'triglycerides': patient.triglycerides
            },
            'conditions': {
                'has_diabetes': patient.has_diabetes,
                'has_hypertension': patient.has_hypertension,
                'has_cardiovascular_disease': patient.has_cardiovascular_disease
            },
            'lifestyle': {
                'smoking_status': patient.smoking_status,
                'exercise_frequency': patient.exercise_frequency,
                'diet_quality': patient.diet_quality,
                'stress_level': patient.stress_level,
                'sleep_hours': patient.sleep_hours,
                'medication_compliance': patient.medication_compliance
            },
            'risk_assessment': {
                'overall_risk': risk_assessment.overall_risk,
                'risk_category': risk_assessment.risk_category,
                'risk_factors': risk_assessment.risk_factors,
                'protective_factors': risk_assessment.protective_factors
            },
            'recommendations': [
                {
                    'category': rec.category,
                    'priority': rec.priority,
                    'recommendation': rec.recommendation,
                    'rationale': rec.rationale,
                    'confidence': rec.confidence,
                    'evidence_level': rec.evidence_level,
                    'actionable': rec.actionable
                }
                for rec in risk_assessment.recommendations
            ],
            'timestamp': datetime.now().isoformat(),
            'disclaimer': 'This summary is for research purposes only. Not for clinical use.'
        }
