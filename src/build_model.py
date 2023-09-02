"""
build_data Module

This module is used for creating the models.

References: 
    1. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

from torchvision import models
import torch.nn as nn

def initialize_model():
    # Load pretrained model params
    model = models.resnet50(pretrained=True)

    # Replace the original classifier with a new Linear layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Ensure all params get updated during finetuning
    for param in model.parameters():
        param.requires_grad = True
    return model

