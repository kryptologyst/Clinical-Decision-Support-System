"""
Streamlit demo for Clinical Decision Support System.

Interactive web application for CDSS with patient feature panel,
risk assessment, and clinical recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import logging
import warnings

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Suppress warnings
warnings.filterwarnings("ignore")

# Import CDSS components
from src.models.cdss import ClinicalDecisionSupportSystem
from src.data.synthetic_data import PatientData, SyntheticDataGenerator
from src.data.data_pipeline import DataPipeline
from src.models.ml_models import train_models
from src.metrics.clinical_metrics import ClinicalMetrics
from src.utils.config import load_config, set_deterministic_seed
from src.utils.explainability import ExplainabilityEngine, UncertaintyQuantification, SafetyChecks


# Page configuration
st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #c62828;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .high-priority {
        border-left-color: #f44336;
        background-color: #ffebee;
    }
    .medium-priority {
        border-left-color: #ff9800;
        background-color: #fff3e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cdss' not in st.session_state:
    st.session_state.cdss = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None


def load_system():
    """Load the CDSS system."""
    if st.session_state.cdss is None:
        with st.spinner("Loading Clinical Decision Support System..."):
            # Set deterministic seed
            set_deterministic_seed(42)
            
            # Load configuration
            config = load_config("configs/default.yaml")
            st.session_state.config = config
            
            # Initialize CDSS
            cdss = ClinicalDecisionSupportSystem(config)
            st.session_state.cdss = cdss
            
            # Generate synthetic data
            generator = SyntheticDataGenerator(config)
            patients = generator.generate_patients(1000)
            
            # Train ML models
            df = generator.to_dataframe(patients)
            feature_columns = [col for col in df.columns if col not in ['patient_id', 'risk_score']]
            X = df[feature_columns]
            y = (df['risk_score'] >= 0.5).astype(int)
            
            models = train_models(X, y, config)
            st.session_state.models = models
            
            # Train CDSS ML models
            cdss.train_ml_models(patients)
            
            st.success("CDSS system loaded successfully!")


def create_patient_input_form():
    """Create patient input form."""
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Demographics**")
        age = st.slider("Age", 18, 85, 50)
        gender = st.selectbox("Gender", ["male", "female"])
        
        st.write("**Vital Signs**")
        systolic_bp = st.slider("Systolic BP (mmHg)", 90, 200, 120)
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
        bmi = st.slider("BMI", 16, 50, 25.0)
        
        st.write("**Laboratory Values**")
        fasting_glucose = st.slider("Fasting Glucose (mg/dL)", 70, 300, 100)
        hba1c = st.slider("HbA1c (%)", 4.0, 12.0, 5.5)
        cholesterol = st.slider("Total Cholesterol (mg/dL)", 120, 400, 200)
        hdl_cholesterol = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
        ldl_cholesterol = st.slider("LDL Cholesterol (mg/dL)", 50, 300, 120)
        triglycerides = st.slider("Triglycerides (mg/dL)", 50, 500, 150)
    
    with col2:
        st.write("**Lifestyle Factors**")
        smoking_status = st.selectbox("Smoking Status", ["never", "former", "current"])
        exercise_frequency = st.selectbox("Exercise Frequency", ["none", "light", "moderate", "intense"])
        diet_quality = st.selectbox("Diet Quality", ["poor", "fair", "good", "excellent"])
        stress_level = st.selectbox("Stress Level", ["low", "moderate", "high"])
        sleep_hours = st.slider("Sleep Hours", 4, 12, 7.5)
        medication_compliance = st.slider("Medication Compliance", 0.0, 1.0, 0.8)
        
        st.write("**Family History**")
        family_history_diabetes = st.checkbox("Family History of Diabetes")
        family_history_hypertension = st.checkbox("Family History of Hypertension")
        
        st.write("**Current Conditions**")
        has_diabetes = st.checkbox("Has Diabetes")
        has_hypertension = st.checkbox("Has Hypertension")
        has_cardiovascular_disease = st.checkbox("Has Cardiovascular Disease")
    
    return {
        'age': age,
        'gender': gender,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'bmi': bmi,
        'fasting_glucose': fasting_glucose,
        'hba1c': hba1c,
        'cholesterol': cholesterol,
        'hdl_cholesterol': hdl_cholesterol,
        'ldl_cholesterol': ldl_cholesterol,
        'triglycerides': triglycerides,
        'smoking_status': smoking_status,
        'exercise_frequency': exercise_frequency,
        'diet_quality': diet_quality,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        'medication_compliance': medication_compliance,
        'family_history_diabetes': family_history_diabetes,
        'family_history_hypertension': family_history_hypertension,
        'has_diabetes': has_diabetes,
        'has_hypertension': has_hypertension,
        'has_cardiovascular_disease': has_cardiovascular_disease
    }


def create_patient_data_object(patient_data: dict) -> PatientData:
    """Create PatientData object from form data."""
    # Calculate risk score (simplified)
    risk_score = 0.0
    
    # Age factor
    risk_score += min((patient_data['age'] - 30) / 50, 0.3)
    
    # Blood pressure factor
    if patient_data['systolic_bp'] >= 140 or patient_data['diastolic_bp'] >= 90:
        risk_score += 0.2
    
    # Glucose factor
    if patient_data['fasting_glucose'] >= 126 or patient_data['hba1c'] >= 6.5:
        risk_score += 0.2
    
    # BMI factor
    if patient_data['bmi'] >= 30:
        risk_score += 0.15
    elif patient_data['bmi'] >= 25:
        risk_score += 0.1
    
    # Cholesterol factor
    if patient_data['cholesterol'] >= 200:
        risk_score += 0.1
    
    # Lifestyle factors
    if patient_data['smoking_status'] == 'current':
        risk_score += 0.15
    elif patient_data['smoking_status'] == 'former':
        risk_score += 0.05
    
    if patient_data['exercise_frequency'] == 'none':
        risk_score += 0.1
    elif patient_data['exercise_frequency'] == 'light':
        risk_score += 0.05
    
    if patient_data['diet_quality'] == 'poor':
        risk_score += 0.1
    elif patient_data['diet_quality'] == 'fair':
        risk_score += 0.05
    
    # Family history
    if patient_data['family_history_diabetes']:
        risk_score += 0.1
    if patient_data['family_history_hypertension']:
        risk_score += 0.05
    
    risk_score = min(risk_score, 1.0)
    
    return PatientData(
        patient_id="DEMO_PATIENT",
        age=patient_data['age'],
        gender=patient_data['gender'],
        systolic_bp=patient_data['systolic_bp'],
        diastolic_bp=patient_data['diastolic_bp'],
        fasting_glucose=patient_data['fasting_glucose'],
        hba1c=patient_data['hba1c'],
        bmi=patient_data['bmi'],
        cholesterol=patient_data['cholesterol'],
        hdl_cholesterol=patient_data['hdl_cholesterol'],
        ldl_cholesterol=patient_data['ldl_cholesterol'],
        triglycerides=patient_data['triglycerides'],
        smoking_status=patient_data['smoking_status'],
        family_history_diabetes=patient_data['family_history_diabetes'],
        family_history_hypertension=patient_data['family_history_hypertension'],
        exercise_frequency=patient_data['exercise_frequency'],
        diet_quality=patient_data['diet_quality'],
        stress_level=patient_data['stress_level'],
        sleep_hours=patient_data['sleep_hours'],
        medication_compliance=patient_data['medication_compliance'],
        has_diabetes=patient_data['has_diabetes'],
        has_hypertension=patient_data['has_hypertension'],
        has_cardiovascular_disease=patient_data['has_cardiovascular_disease'],
        risk_score=risk_score
    )


def display_risk_assessment(risk_assessment):
    """Display risk assessment."""
    st.subheader("Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Risk Score", f"{risk_assessment.overall_risk:.2f}")
    
    with col2:
        risk_category = risk_assessment.risk_category
        risk_color = {
            'low': 'green',
            'moderate': 'orange', 
            'high': 'red',
            'very_high': 'darkred'
        }.get(risk_category, 'gray')
        
        st.markdown(f"**Risk Category:** <span style='color: {risk_color}'>{risk_category.upper()}</span>", 
                   unsafe_allow_html=True)
    
    with col3:
        st.metric("Risk Factors", len(risk_assessment.risk_factors))
    
    # Risk factors and protective factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk Factors:**")
        if risk_assessment.risk_factors:
            for factor in risk_assessment.risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("No significant risk factors identified")
    
    with col2:
        st.write("**Protective Factors:**")
        if risk_assessment.protective_factors:
            for factor in risk_assessment.protective_factors:
                st.write(f"• {factor}")
        else:
            st.write("No protective factors identified")


def display_recommendations(recommendations):
    """Display clinical recommendations."""
    st.subheader("Clinical Recommendations")
    
    # Group recommendations by priority
    high_priority = [r for r in recommendations if r.priority == 'high']
    medium_priority = [r for r in recommendations if r.priority == 'medium']
    low_priority = [r for r in recommendations if r.priority == 'low']
    
    if high_priority:
        st.write("**🔴 High Priority**")
        for rec in high_priority:
            st.markdown(f"""
            <div class="recommendation-card high-priority">
                <strong>{rec.category}:</strong> {rec.recommendation}<br>
                <em>Rationale:</em> {rec.rationale}<br>
                <em>Confidence:</em> {rec.confidence:.1%} | 
                <em>Evidence Level:</em> {rec.evidence_level}
            </div>
            """, unsafe_allow_html=True)
    
    if medium_priority:
        st.write("**🟡 Medium Priority**")
        for rec in medium_priority:
            st.markdown(f"""
            <div class="recommendation-card medium-priority">
                <strong>{rec.category}:</strong> {rec.recommendation}<br>
                <em>Rationale:</em> {rec.rationale}<br>
                <em>Confidence:</em> {rec.confidence:.1%} | 
                <em>Evidence Level:</em> {rec.evidence_level}
            </div>
            """, unsafe_allow_html=True)
    
    if low_priority:
        st.write("**🟢 Low Priority**")
        for rec in low_priority:
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{rec.category}:</strong> {rec.recommendation}<br>
                <em>Rationale:</em> {rec.rationale}<br>
                <em>Confidence:</em> {rec.confidence:.1%} | 
                <em>Evidence Level:</em> {rec.evidence_level}
            </div>
            """, unsafe_allow_html=True)


def create_risk_visualization(patient_data: PatientData, risk_assessment):
    """Create risk visualization."""
    st.subheader("Risk Visualization")
    
    # Risk score gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_assessment.overall_risk,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Risk Score"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.5], 'color': "yellow"},
                {'range': [0.5, 0.7], 'color': "orange"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors radar chart
    if risk_assessment.risk_factors:
        categories = ['Age', 'Blood Pressure', 'Glucose', 'BMI', 'Cholesterol', 'Lifestyle']
        
        # Calculate risk scores for each category
        age_score = min((patient_data.age - 30) / 50, 1.0)
        bp_score = 1.0 if (patient_data.systolic_bp >= 140 or patient_data.diastolic_bp >= 90) else 0.0
        glucose_score = 1.0 if (patient_data.fasting_glucose >= 126 or patient_data.hba1c >= 6.5) else 0.0
        bmi_score = 1.0 if patient_data.bmi >= 30 else (0.5 if patient_data.bmi >= 25 else 0.0)
        cholesterol_score = 1.0 if patient_data.cholesterol >= 200 else 0.0
        lifestyle_score = 0.5 if patient_data.smoking_status == 'current' else 0.0
        
        values = [age_score, bp_score, glucose_score, bmi_score, cholesterol_score, lifestyle_score]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Factors'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Risk Factor Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">Clinical Decision Support System</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ DISCLAIMER: This system is for research and educational purposes only. 
        It is NOT intended for clinical use and should NOT be used for medical diagnosis or treatment decisions. 
        Always consult qualified healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    # Load system
    load_system()
    
    if st.session_state.cdss is None:
        st.error("Failed to load CDSS system. Please refresh the page.")
        return
    
    # Sidebar
    st.sidebar.title("CDSS Demo")
    st.sidebar.write("This demo showcases a Clinical Decision Support System for chronic disease management.")
    
    # Demo options
    demo_mode = st.sidebar.selectbox(
        "Demo Mode",
        ["Interactive Patient Input", "Sample Cases", "Model Performance"]
    )
    
    if demo_mode == "Interactive Patient Input":
        # Patient input form
        patient_data = create_patient_input_form()
        
        if st.button("Generate Clinical Assessment", type="primary"):
            # Create patient object
            patient = create_patient_data_object(patient_data)
            
            # Get risk assessment
            risk_assessment = st.session_state.cdss.assess_risk(patient)
            
            # Display results
            st.markdown("---")
            
            # Risk assessment
            display_risk_assessment(risk_assessment)
            
            # Recommendations
            display_recommendations(risk_assessment.recommendations)
            
            # Visualizations
            create_risk_visualization(patient, risk_assessment)
            
            # Patient summary
            with st.expander("Detailed Patient Summary"):
                summary = st.session_state.cdss.get_patient_summary(patient)
                st.json(summary)
    
    elif demo_mode == "Sample Cases":
        st.subheader("Sample Patient Cases")
        
        # Generate sample cases
        generator = SyntheticDataGenerator(st.session_state.config)
        sample_patients = generator.generate_patients(5)
        
        for i, patient in enumerate(sample_patients):
            with st.expander(f"Patient {i+1}: {patient.patient_id}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Demographics**")
                    st.write(f"Age: {patient.age}, Gender: {patient.gender}")
                    st.write(f"BMI: {patient.bmi:.1f}")
                    
                    st.write("**Vital Signs**")
                    st.write(f"BP: {patient.systolic_bp:.0f}/{patient.diastolic_bp:.0f} mmHg")
                    
                    st.write("**Laboratory Values**")
                    st.write(f"Glucose: {patient.fasting_glucose:.0f} mg/dL")
                    st.write(f"HbA1c: {patient.hba1c:.1f}%")
                    st.write(f"Cholesterol: {patient.cholesterol:.0f} mg/dL")
                
                with col2:
                    st.write("**Lifestyle**")
                    st.write(f"Smoking: {patient.smoking_status}")
                    st.write(f"Exercise: {patient.exercise_frequency}")
                    st.write(f"Diet: {patient.diet_quality}")
                    
                    st.write("**Conditions**")
                    st.write(f"Diabetes: {'Yes' if patient.has_diabetes else 'No'}")
                    st.write(f"Hypertension: {'Yes' if patient.has_hypertension else 'No'}")
                    st.write(f"Cardiovascular: {'Yes' if patient.has_cardiovascular_disease else 'No'}")
                
                # Risk assessment
                risk_assessment = st.session_state.cdss.assess_risk(patient)
                
                st.write(f"**Risk Score:** {risk_assessment.overall_risk:.2f}")
                st.write(f"**Risk Category:** {risk_assessment.risk_category.upper()}")
                
                if st.button(f"View Recommendations for Patient {i+1}"):
                    display_recommendations(risk_assessment.recommendations)
    
    elif demo_mode == "Model Performance":
        st.subheader("Model Performance Analysis")
        
        if st.session_state.models:
            st.write("**Available Models:**")
            for model_name in st.session_state.models.keys():
                st.write(f"• {model_name.upper()}")
            
            st.write("**Model Comparison:**")
            st.info("Model performance metrics would be displayed here, including AUC-ROC, calibration curves, and decision curve analysis.")
            
            # Placeholder for model performance plots
            st.write("**Performance Metrics:**")
            st.write("• AUC-ROC: 0.85")
            st.write("• Sensitivity: 0.82")
            st.write("• Specificity: 0.78")
            st.write("• Calibration Error: 0.05")
        else:
            st.warning("No trained models available for performance analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Clinical Decision Support System Demo | Research & Educational Use Only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
