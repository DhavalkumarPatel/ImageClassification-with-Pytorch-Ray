"""
utils Module

This module is defined to read the configuration file and set mlflow URI

References: 
    1. https://github.com/GokuMohandas/Made-With-ML
"""

from typing import Dict, Tuple
import os
import yaml
from pathlib import Path

def read_config() -> Tuple[Dict, Dict, Dict]:
    """Function to read the configuration file."""
    
    # The function assumes that the config file is stored in the same dir.
    script_directory = os.path.dirname(__file__)
    file_path = os.path.join(script_directory, "config.YAML")
    
    # Read the configs
    with open(file_path, "r") as config_file:
        configs = yaml.safe_load(config_file)
    
    train_loop_config = configs["train_loop_config"]
    scaling_config_read = configs["scaling_config"]
    mlflow_config = configs["mlflow_config"]
        
    return train_loop_config, scaling_config_read, mlflow_config

def configure_mlflow(mlflow_config) -> str:
    """Function to set the mlflow training uri."""
    
    MODEL_REGISTRY = Path(mlflow_config["mlflow_path"])
    Path(MODEL_REGISTRY).mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
      
    return MLFLOW_TRACKING_URI