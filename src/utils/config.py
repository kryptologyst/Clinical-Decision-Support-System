"""
Configuration management for the Clinical Decision Support System.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from omegaconf import OmegaConf


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """Get the best available device with fallback.
    
    Returns:
        torch.device: CUDA, MPS (Apple Silicon), or CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict containing configuration parameters
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Create default config if it doesn't exist
        default_config = get_default_config()
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        return default_config
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the CDSS.
    
    Returns:
        Dict containing default configuration parameters
    """
    return {
        'system': {
            'name': 'Clinical Decision Support System',
            'version': '1.0.0',
            'description': 'Research demo for chronic disease management',
            'disclaimer': 'For research purposes only. Not for clinical use.'
        },
        'data': {
            'n_patients': 1000,
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42,
            'patient_level_split': True
        },
        'models': {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'tabnet': {
                'n_d': 64,
                'n_a': 64,
                'n_steps': 5,
                'gamma': 1.5,
                'lambda_sparse': 1e-3,
                'optimizer_fn': 'adam',
                'optimizer_params': {'lr': 2e-2},
                'scheduler_params': {'step_size': 50, 'gamma': 0.9},
                'scheduler_fn': 'step',
                'mask_type': 'entmax'
            }
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        },
        'evaluation': {
            'metrics': ['auc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'npv'],
            'calibration_bins': 10,
            'decision_threshold': 0.5
        },
        'clinical_rules': {
            'hypertension': {
                'systolic_threshold': 140,
                'diastolic_threshold': 90
            },
            'diabetes': {
                'glucose_threshold': 126,
                'hba1c_threshold': 6.5
            },
            'cardiovascular': {
                'cholesterol_threshold': 200,
                'bmi_threshold': 30
            },
            'preventive_care': {
                'colonoscopy_age': 50,
                'mammography_age': 40
            }
        },
        'device': str(get_device()),
        'seed': 42
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values.
    
    Args:
        config: Original configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    def deep_update(d: Dict, u: Dict) -> Dict:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return deep_update(config.copy(), updates)
