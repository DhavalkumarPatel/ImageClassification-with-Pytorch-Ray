"""
build_data Module

This module is used for creating the train and val datasets.

References: 
    1. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

# Required Imports
import os
import shutil
from typing import Dict

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import read_config

def define_transforms() -> Dict[str, DataLoader]:
    """Transform Definitions for the train and val set."""
    
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms

def build_datasets_download() -> Dict[str, datasets.ImageFolder]:
    """Function to download the datasets (online)."""
    
    # Get the tranforms to be used.
    data_transforms = define_transforms()
    
    # Download and Unzip the data for the current worker.
    os.system(
        "wget https://download.pytorch.org/tutorial/hymenoptera_data.zip >/dev/null 2>&1"
    )
    os.system("unzip hymenoptera_data.zip >/dev/null 2>&1")
    
    # Build and Return the train and val datasets.
    torch_datasets = {}
    for split in ["train", "val"]:
        torch_datasets[split] = datasets.ImageFolder(
            os.path.join("./hymenoptera_data", split), data_transforms[split]
        )
    return torch_datasets

def build_datasets_local() -> Dict[str, datasets.ImageFolder]:
    """Function to use the data stored locally."""
    
    # Read the config file to extract the data path
    train_loop_config, scaling_config_read, mlflow_config = read_config() 
    CENTRAL_STORAGE = mlflow_config["data_path"]
     
    # Get the tranforms to be used. 
    data_transforms = define_transforms()
    
    # Copy the data to the current worker.
    ray_dir = os.getcwd() 
    shutil.copytree(CENTRAL_STORAGE, ray_dir, dirs_exist_ok=True)
    
    # Build and Return the train and val datasets.
    torch_datasets = {}
    for split in ["train", "val"]:
        torch_datasets[split] = datasets.ImageFolder(
            os.path.join(ray_dir, split), data_transforms[split]
        )
    return torch_datasets
    
def build_datasets_test() -> datasets.ImageFolder:
    """Function to prepare the data for evalution."""
    
    # Read the config file to extract the data path
    train_loop_config, scaling_config_read, mlflow_config = read_config() 
    CENTRAL_STORAGE = mlflow_config["data_path"]
    
    # Get the tranforms to be used. 
    data_transforms = define_transforms()
    
    # Locate the evaluation datasets. This can be changed to test folder.
    test_directory = CENTRAL_STORAGE + "/val"
    
    return datasets.ImageFolder(test_directory, data_transforms["val"])
