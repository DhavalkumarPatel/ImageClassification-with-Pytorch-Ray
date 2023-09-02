# Module for implementing the model training

# Load the required functions and modules
from build_data import build_datasets_local
from build_model import initialize_model
from train_utils import train_step

# Required Imports
import os
import yaml
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import ray
import ray.train as train
from ray.air import session

from ray.train.torch import TorchTrainer, TorchCheckpoint
from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig

from typing import Dict

# Define the train_loop_per_worker for TorchTrainer.
def train_loop_per_worker(configs: Dict[str, float]) -> None:
    """ This function implements training loop across multiple workers. 
    
    Args:
        configs: A Dictionary containing the parameters used in training.
    
    Returns: 
        None
    """
    
    import warnings
    warnings.filterwarnings("ignore")

    # Calculate the batch size for a single worker
    worker_batch_size = configs["batch_size"] // session.get_world_size()

    # Build datasets on each worker
    torch_datasets = build_datasets_local()

    # Prepare dataloader for each worker
    dataloaders = dict()
    dataloaders["train"] = DataLoader(
        torch_datasets["train"], batch_size=worker_batch_size, shuffle=True
    )
    dataloaders["val"] = DataLoader(
        torch_datasets["val"], batch_size=worker_batch_size, shuffle=False
    )

    # Distribute
    dataloaders["train"] = train.torch.prepare_data_loader(dataloaders["train"])
    dataloaders["val"] = train.torch.prepare_data_loader(dataloaders["val"])
    
    # Get the current device
    device = train.torch.get_device()

    # Prepare DDP Model, optimizer, and loss function
    model = initialize_model()
    model = train.torch.prepare_model(model)
    
    # Define the optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=configs["lr"], momentum=configs["momentum"]
    )
    
    # Define the criterion
    criterion = nn.CrossEntropyLoss()

    # Start training loops
    for epoch in range(configs["num_epochs"]):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
                
            # Run the train_step/val step
            running_loss, running_corrects = train_step(dataloaders, 
                                                        phase, 
                                                        model, 
                                                        device, 
                                                        optimizer, 
                                                        criterion)
            
            # Calculate the loss and accuracy
            size = len(torch_datasets[phase]) // session.get_world_size()
            epoch_loss = running_loss / size
            epoch_acc = running_corrects / size

            if session.get_world_rank() == 0:
                print(
                    "Epoch {}-{} Loss: {:.4f} Acc: {:.4f}".format(
                        epoch, phase, epoch_loss, epoch_acc
                    )
                )

            # Report metrics and checkpoint every epoch
            if phase == "val":
                checkpoint = TorchCheckpoint.from_dict(
                    {
                        "epoch": epoch,
                        "model": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                )
                session.report(
                    metrics={"val_loss": epoch_loss, "val_acc": epoch_acc},
                    checkpoint=checkpoint,
                )


def read_config():
    
    script_directory = os.path.dirname(__file__)
    file_path = os.path.join(script_directory, "config.YAML")
    
    with open(file_path, "r") as config_file:
        configs = yaml.safe_load(config_file)
    
    train_loop_config = configs["train_loop_config"]
    scaling_config_read = configs["scaling_config"]
    run_config_read = configs["run_config"]
    
    return train_loop_config, scaling_config_read, run_config_read


def torch_trainer():
    """ The entry point of the training process using Ray. """
    
    # Read the configurations from the YAML file.
    train_loop_config, scaling_config_read, run_config_read = read_config() 
    
    # Scale out model training.
    scaling_config = ScalingConfig(
        **scaling_config_read)
    
    # Save the latest checkpoint
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute ="val_acc",
        checkpoint_score_order = "max")
    
    # Set experiment name and checkpoint configs
    run_config = RunConfig(
        **run_config_read, 
        checkpoint_config=checkpoint_config,
    )
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    result = trainer.fit()
    print(result)
    
if __name__ == "__main__": 
    if ray.is_initialized():
        ray.shutdown()
    torch_trainer()