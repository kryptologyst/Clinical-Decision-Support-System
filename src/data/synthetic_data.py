"""
Synthetic data generation for Clinical Decision Support System.

Generates realistic synthetic patient data for hypertension and diabetes management
while ensuring no PHI/PII is included.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random


@dataclass
class PatientData:
    """Patient data structure for CDSS."""
    patient_id: str
    age: int
    gender: str
    systolic_bp: float
    diastolic_bp: float
    fasting_glucose: float
    hba1c: float
    bmi: float
    cholesterol: float
    hdl_cholesterol: float
    ldl_cholesterol: float
    triglycerides: float
    smoking_status: str
    family_history_diabetes: bool
    family_history_hypertension: bool
    exercise_frequency: str
    diet_quality: str
    stress_level: str
    sleep_hours: float
    medication_compliance: float
    has_diabetes: bool
    has_hypertension: bool
    has_cardiovascular_disease: bool
    risk_score: float


class SyntheticDataGenerator:
    """Generate synthetic patient data for CDSS research."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.random_state = config.get('data', {}).get('random_state', 42)
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        # Clinical thresholds
        self.clinical_rules = config.get('clinical_rules', {})
        
    def generate_patient(self, patient_id: str) -> PatientData:
        """Generate a single synthetic patient.
        
        Args:
            patient_id: Unique patient identifier
            
        Returns:
            PatientData object with synthetic patient information
        """
        # Basic demographics
        age = np.random.randint(18, 85)
        gender = np.random.choice(['male', 'female'])
        
        # Generate correlated clinical parameters
        # BMI affects blood pressure and glucose
        bmi = np.random.normal(26, 5)
        bmi = max(16, min(50, bmi))  # Clamp to realistic range
        
        # Blood pressure (correlated with age, BMI, gender)
        bp_base = 120 + (age - 40) * 0.5 + (bmi - 25) * 1.2
        if gender == 'male':
            bp_base += 5
        systolic_bp = np.random.normal(bp_base, 15)
        diastolic_bp = np.random.normal(systolic_bp * 0.6, 8)
        
        # Glucose levels (correlated with BMI, age)
        glucose_base = 90 + (bmi - 25) * 2 + (age - 40) * 0.3
        fasting_glucose = np.random.normal(glucose_base, 20)
        
        # HbA1c (correlated with glucose)
        hba1c = 4.5 + (fasting_glucose - 100) * 0.02 + np.random.normal(0, 0.5)
        
        # Lipid profile
        cholesterol = np.random.normal(200, 40)
        hdl_cholesterol = np.random.normal(50, 15)
        ldl_cholesterol = cholesterol - hdl_cholesterol - np.random.normal(20, 10)
        triglycerides = np.random.normal(150, 80)
        
        # Lifestyle factors
        smoking_status = np.random.choice(['never', 'former', 'current'], p=[0.6, 0.25, 0.15])
        exercise_frequency = np.random.choice(['none', 'light', 'moderate', 'intense'], 
                                            p=[0.2, 0.3, 0.35, 0.15])
        diet_quality = np.random.choice(['poor', 'fair', 'good', 'excellent'], 
                                      p=[0.15, 0.35, 0.35, 0.15])
        stress_level = np.random.choice(['low', 'moderate', 'high'], p=[0.3, 0.5, 0.2])
        sleep_hours = np.random.normal(7.5, 1.5)
        
        # Family history
        family_history_diabetes = np.random.choice([True, False], p=[0.3, 0.7])
        family_history_hypertension = np.random.choice([True, False], p=[0.4, 0.6])
        
        # Medication compliance
        medication_compliance = np.random.beta(2, 1)  # Skewed towards higher compliance
        
        # Determine conditions based on clinical thresholds
        htn_thresholds = self.clinical_rules.get('hypertension', {})
        diabetes_thresholds = self.clinical_rules.get('diabetes', {})
        cv_thresholds = self.clinical_rules.get('cardiovascular', {})
        
        has_hypertension = (systolic_bp >= htn_thresholds.get('systolic_threshold', 140) or 
                          diastolic_bp >= htn_thresholds.get('diastolic_threshold', 90))
        
        has_diabetes = (fasting_glucose >= diabetes_thresholds.get('glucose_threshold', 126) or
                       hba1c >= diabetes_thresholds.get('hba1c_threshold', 6.5))
        
        has_cardiovascular_disease = (cholesterol >= cv_thresholds.get('cholesterol_threshold', 200) or
                                    bmi >= cv_thresholds.get('bmi_threshold', 30))
        
        # Calculate risk score (simplified)
        risk_score = self._calculate_risk_score(
            age, systolic_bp, diastolic_bp, fasting_glucose, hba1c, bmi, 
            cholesterol, smoking_status, family_history_diabetes, 
            family_history_hypertension, exercise_frequency, diet_quality
        )
        
        return PatientData(
            patient_id=patient_id,
            age=age,
            gender=gender,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            fasting_glucose=fasting_glucose,
            hba1c=hba1c,
            bmi=bmi,
            cholesterol=cholesterol,
            hdl_cholesterol=hdl_cholesterol,
            ldl_cholesterol=ldl_cholesterol,
            triglycerides=triglycerides,
            smoking_status=smoking_status,
            family_history_diabetes=family_history_diabetes,
            family_history_hypertension=family_history_hypertension,
            exercise_frequency=exercise_frequency,
            diet_quality=diet_quality,
            stress_level=stress_level,
            sleep_hours=sleep_hours,
            medication_compliance=medication_compliance,
            has_diabetes=has_diabetes,
            has_hypertension=has_hypertension,
            has_cardiovascular_disease=has_cardiovascular_disease,
            risk_score=risk_score
        )
    
    def _calculate_risk_score(self, age: int, systolic_bp: float, diastolic_bp: float,
                            fasting_glucose: float, hba1c: float, bmi: float,
                            cholesterol: float, smoking_status: str,
                            family_history_diabetes: bool, family_history_hypertension: bool,
                            exercise_frequency: str, diet_quality: str) -> float:
        """Calculate a simplified cardiovascular risk score.
        
        Args:
            Various patient parameters
            
        Returns:
            Risk score between 0 and 1
        """
        score = 0.0
        
        # Age factor
        score += min((age - 30) / 50, 0.3)
        
        # Blood pressure factor
        if systolic_bp >= 140 or diastolic_bp >= 90:
            score += 0.2
        
        # Glucose factor
        if fasting_glucose >= 126 or hba1c >= 6.5:
            score += 0.2
        
        # BMI factor
        if bmi >= 30:
            score += 0.15
        elif bmi >= 25:
            score += 0.1
        
        # Cholesterol factor
        if cholesterol >= 200:
            score += 0.1
        
        # Lifestyle factors
        if smoking_status == 'current':
            score += 0.15
        elif smoking_status == 'former':
            score += 0.05
        
        if exercise_frequency == 'none':
            score += 0.1
        elif exercise_frequency == 'light':
            score += 0.05
        
        if diet_quality == 'poor':
            score += 0.1
        elif diet_quality == 'fair':
            score += 0.05
        
        # Family history
        if family_history_diabetes:
            score += 0.1
        if family_history_hypertension:
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def generate_patients(self, n_patients: int) -> List[PatientData]:
        """Generate multiple synthetic patients.
        
        Args:
            n_patients: Number of patients to generate
            
        Returns:
            List of PatientData objects
        """
        patients = []
        for i in range(n_patients):
            patient_id = f"PAT_{i+1:06d}"
            patient = self.generate_patient(patient_id)
            patients.append(patient)
        
        return patients
    
    def to_dataframe(self, patients: List[PatientData]) -> pd.DataFrame:
        """Convert list of patients to pandas DataFrame.
        
        Args:
            patients: List of PatientData objects
            
        Returns:
            DataFrame with patient data
        """
        data = []
        for patient in patients:
            data.append({
                'patient_id': patient.patient_id,
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
                'has_cardiovascular_disease': patient.has_cardiovascular_disease,
                'risk_score': patient.risk_score
            })
        
        return pd.DataFrame(data)


def generate_synthetic_patients(n_patients: int, config: Dict[str, Any]) -> List[PatientData]:
    """Convenience function to generate synthetic patients.
    
    Args:
        n_patients: Number of patients to generate
        config: Configuration dictionary
        
    Returns:
        List of PatientData objects
    """
    generator = SyntheticDataGenerator(config)
    return generator.generate_patients(n_patients)
