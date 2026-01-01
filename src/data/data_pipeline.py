"""
Data pipelines for Clinical Decision Support System.

Handles data loading, preprocessing, splitting, and feature engineering
for EHR/tabular data with proper patient-level splits and imbalance handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
import logging

from ..data.synthetic_data import PatientData, SyntheticDataGenerator


class DataPipeline:
    """Data pipeline for CDSS with proper preprocessing and splitting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data configuration
        self.data_config = config.get('data', {})
        self.test_size = self.data_config.get('test_size', 0.2)
        self.val_size = self.data_config.get('val_size', 0.2)
        self.random_state = self.data_config.get('random_state', 42)
        self.patient_level_split = self.data_config.get('patient_level_split', True)
        
        # Preprocessing components
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.target_name = None
        
        # Data splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Patient IDs for tracking
        self.train_patient_ids = None
        self.val_patient_ids = None
        self.test_patient_ids = None
        
    def load_synthetic_data(self, n_patients: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load synthetic patient data.
        
        Args:
            n_patients: Number of patients to generate
            
        Returns:
            Tuple of (features, targets)
        """
        self.logger.info(f"Generating {n_patients} synthetic patients...")
        
        generator = SyntheticDataGenerator(self.config)
        patients = generator.generate_patients(n_patients)
        
        # Convert to DataFrame
        df = generator.to_dataframe(patients)
        
        # Separate features and target
        target_column = 'risk_score'
        feature_columns = [col for col in df.columns if col not in ['patient_id', target_column]]
        
        X = df[feature_columns]
        y = df[target_column].values
        
        self.feature_names = feature_columns
        self.target_name = target_column
        
        self.logger.info(f"Generated data: {X.shape[0]} patients, {X.shape[1]} features")
        return X, y
    
    def preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Preprocess features with encoding and scaling.
        
        Args:
            X: Input features
            fit: Whether to fit the preprocessors
            
        Returns:
            Preprocessed features
        """
        X_processed = X.copy()
        
        # Encode categorical variables
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col])
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    try:
                        X_processed[col] = self.label_encoders[col].transform(X_processed[col])
                    except ValueError:
                        # Map unseen categories to most frequent category
                        most_frequent = self.label_encoders[col].classes_[0]
                        X_processed[col] = self.label_encoders[col].transform([most_frequent] * len(X_processed))
        
        # Scale numerical features
        if fit:
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X_processed)
            else:
                X_scaled = X_processed.values
        
        return X_scaled
    
    def split_data(self, X: pd.DataFrame, y: np.ndarray, 
                   patient_ids: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Split data into train/validation/test sets.
        
        Args:
            X: Features
            y: Targets
            patient_ids: Optional patient IDs for patient-level splitting
            
        Returns:
            Dictionary with data splits
        """
        self.logger.info("Splitting data into train/validation/test sets...")
        
        if self.patient_level_split and patient_ids is not None:
            # Patient-level splitting to prevent data leakage
            unique_patients = np.unique(patient_ids)
            
            # First split: separate test set
            train_val_patients, test_patients = train_test_split(
                unique_patients, 
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=None  # We'll stratify later if needed
            )
            
            # Second split: separate train and validation
            train_patients, val_patients = train_test_split(
                train_val_patients,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state
            )
            
            # Create masks for each split
            train_mask = np.isin(patient_ids, train_patients)
            val_mask = np.isin(patient_ids, val_patients)
            test_mask = np.isin(patient_ids, test_patients)
            
            # Apply splits
            self.X_train = X[train_mask]
            self.X_val = X[val_mask]
            self.X_test = X[test_mask]
            self.y_train = y[train_mask]
            self.y_val = y[val_mask]
            self.y_test = y[test_mask]
            
            self.train_patient_ids = patient_ids[train_mask]
            self.val_patient_ids = patient_ids[val_mask]
            self.test_patient_ids = patient_ids[test_mask]
            
        else:
            # Standard random splitting
            # First split: separate test set
            X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Second split: separate train and validation
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state
            )
        
        self.logger.info(f"Train: {len(self.X_train)} samples")
        self.logger.info(f"Validation: {len(self.X_val)} samples")
        self.logger.info(f"Test: {len(self.X_test)} samples")
        
        return {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test
        }
    
    def handle_class_imbalance(self, X: np.ndarray, y: np.ndarray, 
                              method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance in training data.
        
        Args:
            X: Training features
            y: Training targets
            method: Method to use ('smote', 'adasyn', 'undersample', 'smote_tomek')
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        self.logger.info(f"Handling class imbalance using {method}...")
        
        # Check if imbalance exists
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) == 1:
            self.logger.warning("Only one class present, skipping balancing")
            return X, y
        
        imbalance_ratio = counts.max() / counts.min()
        self.logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio < 1.5:  # Not significantly imbalanced
            self.logger.info("Data is not significantly imbalanced, skipping balancing")
            return X, y
        
        # Apply balancing method
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=self.random_state)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        try:
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            self.logger.info(f"Balanced data: {X_balanced.shape[0]} samples")
            return X_balanced, y_balanced
        except Exception as e:
            self.logger.warning(f"Failed to balance data: {e}")
            return X, y
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced data.
        
        Args:
            y: Training targets
            
        Returns:
            Dictionary of class weights
        """
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            return {unique_classes[0]: 1.0}
        
        class_weights = compute_class_weight(
            'balanced', classes=unique_classes, y=y
        )
        
        return dict(zip(unique_classes, class_weights))
    
    def create_feature_groups(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Create logical feature groups for analysis.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping group names to feature lists
        """
        feature_groups = {
            'demographics': ['age', 'gender'],
            'vital_signs': ['systolic_bp', 'diastolic_bp', 'bmi'],
            'laboratory': ['fasting_glucose', 'hba1c', 'cholesterol', 'hdl_cholesterol', 
                         'ldl_cholesterol', 'triglycerides'],
            'lifestyle': ['smoking_status', 'exercise_frequency', 'diet_quality', 
                         'stress_level', 'sleep_hours'],
            'family_history': ['family_history_diabetes', 'family_history_hypertension'],
            'compliance': ['medication_compliance'],
            'conditions': ['has_diabetes', 'has_hypertension', 'has_cardiovascular_disease']
        }
        
        # Filter to only include features that exist in the data
        available_groups = {}
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in X.columns]
            if available_features:
                available_groups[group_name] = available_features
        
        return available_groups
    
    def get_sensitive_attributes(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract sensitive attributes for fairness evaluation.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of sensitive attributes
        """
        sensitive_attrs = {}
        
        # Age groups
        if 'age' in X.columns:
            age_groups = pd.cut(X['age'], bins=[0, 40, 60, 100], labels=['young', 'middle', 'old'])
            sensitive_attrs['age_group'] = age_groups.values
        
        # Gender
        if 'gender' in X.columns:
            sensitive_attrs['gender'] = X['gender'].values
        
        # BMI categories
        if 'bmi' in X.columns:
            bmi_categories = pd.cut(X['bmi'], bins=[0, 25, 30, 100], 
                                   labels=['normal', 'overweight', 'obese'])
            sensitive_attrs['bmi_category'] = bmi_categories.values
        
        return sensitive_attrs
    
    def create_cross_validation_splits(self, X: np.ndarray, y: np.ndarray, 
                                     n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Create cross-validation splits.
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of CV folds
            
        Returns:
            List of (X_train, X_val, y_train, y_val) tuples
        """
        cv_splits = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_cv = X[train_idx]
            X_val_cv = X[val_idx]
            y_train_cv = y[train_idx]
            y_val_cv = y[val_idx]
            
            cv_splits.append((X_train_cv, X_val_cv, y_train_cv, y_val_cv))
        
        return cv_splits
    
    def process_pipeline(self, n_patients: int, balance_data: bool = True, 
                        balance_method: str = 'smote') -> Dict[str, Any]:
        """Complete data processing pipeline.
        
        Args:
            n_patients: Number of patients to generate
            balance_data: Whether to balance the data
            balance_method: Method for balancing
            
        Returns:
            Dictionary with processed data and metadata
        """
        self.logger.info("Starting data processing pipeline...")
        
        # Load synthetic data
        X, y = self.load_synthetic_data(n_patients)
        
        # Create patient IDs for patient-level splitting
        patient_ids = np.arange(len(X))
        
        # Split data
        splits = self.split_data(X, y, patient_ids)
        
        # Preprocess features
        X_train_processed = self.preprocess_features(splits['X_train'], fit=True)
        X_val_processed = self.preprocess_features(splits['X_val'], fit=False)
        X_test_processed = self.preprocess_features(splits['X_test'], fit=False)
        
        # Handle class imbalance in training data
        if balance_data:
            # Convert to binary classification for balancing
            y_train_binary = (splits['y_train'] >= 0.5).astype(int)
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(
                X_train_processed, y_train_binary, method=balance_method
            )
            
            # Convert back to continuous targets
            splits['y_train'] = splits['y_train'][y_train_balanced == 1] * 0.8 + splits['y_train'][y_train_balanced == 0] * 0.2
        else:
            X_train_balanced = X_train_processed
            y_train_balanced = (splits['y_train'] >= 0.5).astype(int)
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train_balanced)
        
        # Create feature groups
        feature_groups = self.create_feature_groups(splits['X_train'])
        
        # Get sensitive attributes
        sensitive_attrs = self.get_sensitive_attributes(splits['X_train'])
        
        # Create cross-validation splits
        cv_splits = self.create_cross_validation_splits(X_train_balanced, y_train_balanced)
        
        self.logger.info("Data processing pipeline completed successfully")
        
        return {
            'X_train': X_train_balanced,
            'X_val': X_val_processed,
            'X_test': X_test_processed,
            'y_train': splits['y_train'],
            'y_val': splits['y_val'],
            'y_test': splits['y_test'],
            'y_train_binary': y_train_balanced,
            'y_val_binary': (splits['y_val'] >= 0.5).astype(int),
            'y_test_binary': (splits['y_test'] >= 0.5).astype(int),
            'class_weights': class_weights,
            'feature_groups': feature_groups,
            'sensitive_attributes': sensitive_attrs,
            'cv_splits': cv_splits,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'train_patient_ids': self.train_patient_ids,
            'val_patient_ids': self.val_patient_ids,
            'test_patient_ids': self.test_patient_ids
        }
    
    def save_processed_data(self, processed_data: Dict[str, Any], save_path: str) -> None:
        """Save processed data to disk.
        
        Args:
            processed_data: Processed data dictionary
            save_path: Path to save the data
        """
        import joblib
        
        # Remove non-serializable objects
        save_data = processed_data.copy()
        if 'cv_splits' in save_data:
            del save_data['cv_splits']  # Can be recreated
        
        joblib.dump(save_data, save_path)
        self.logger.info(f"Processed data saved to {save_path}")
    
    def load_processed_data(self, load_path: str) -> Dict[str, Any]:
        """Load processed data from disk.
        
        Args:
            load_path: Path to load the data from
            
        Returns:
            Processed data dictionary
        """
        import joblib
        
        processed_data = joblib.load(load_path)
        self.logger.info(f"Processed data loaded from {load_path}")
        
        return processed_data
