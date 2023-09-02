"""
train_utils Module

This module contains functions used in the training process.

References: 
    1. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

# Imports
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple

# Functions
def evaluate(
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> int:
    """ Function to evaluate the performance of the model.
     
    Args:
        logits: The predictions from the model.
        labels: The true labels for the given input.
        
    Retuns:
        corrects: The number of correct classifications.
    """
    _, preds = torch.max(logits, 1)
    corrects = torch.sum(preds == labels).item()
    return corrects

def train_step(
        dataloaders: Dict[str, DataLoader], 
        phase: str, 
        model: torch.nn.Module, 
        device: torch.device, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module
    ) -> Tuple[float, int]:
    """
    

    Args:
        dataloaders : A dictionary where keys are strings indicating the different 
            training phase (i.e., train, val) and the corresponding values are 
            the respective dataloaders.
        phase : This argument indicates the state of the training process 
            (i.e., train or val).
        model : torch.nn.Module
        device : torch.device
        optimizer : torch.optim.Optimizer 
        criterion : torch.nn.Module 

    Returns:
        The function returns the loss and number of correct classifications.

    """
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(phase == "train"):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            if phase == "train":
                loss.backward()
                optimizer.step()

        # calculate statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += evaluate(outputs, labels)
    
    return running_loss, running_corrects

