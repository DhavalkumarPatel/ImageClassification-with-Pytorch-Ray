from ray.train.torch import TorchCheckpoint
from build_model import initialize_model
from build_data import build_datasets_test
from torch.utils.data import DataLoader
from train_utils import evaluate

import os
import torch

def initialize_model_from_uri(checkpoint_uri):
    checkpoint = TorchCheckpoint.from_directory(checkpoint_uri)
    resnet50 = initialize_model()
    return checkpoint.get_model(model=resnet50)

CHECKPOINT_URI = "/home/dhaval/Projects/NewRay/results/ray_results/finetune-resnet/TorchTrainer_3d3f1_00000_0_2023-08-24_12-35-30/checkpoint_000009/"

model = initialize_model_from_uri(CHECKPOINT_URI)
device = torch.device("cuda")      

model = model.to(device)
model.eval()

test_dataset = build_datasets_test(os.getcwd())
dataloader = DataLoader(test_dataset, batch_size=32, num_workers=2)
corrects = 0
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    preds = model(inputs)
    corrects += evaluate(preds, labels)

print("Accuracy: ", corrects / len(dataloader.dataset))

