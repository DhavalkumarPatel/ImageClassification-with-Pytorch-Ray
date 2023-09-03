"""
evaluate_engine Module

This module performs model evaluation.

Author: Dhaval Patel
"""

from urllib.parse import urlparse
from utils import configure_mlflow, read_config
from ray.air import Result
from ray.train.torch import TorchPredictor
from data_utils import build_datasets_test
from build_model import initialize_model
from train_utils import evaluate_numpy

from torch.utils.data import DataLoader

import mlflow

import typer
from typing_extensions import Annotated

# Initialize Typer CLI app
app = typer.Typer()

def get_best_checkpoint(experiment_name):
    """Function to return the best checkpoint for the given run."""
    
    # Read the configuration file for MLFLOW config
    train_loop_config, scaling_config_read, mlflow_config = read_config() 
    MLFLOW_TRACKING_URI = configure_mlflow(mlflow_config)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Search for the best checkpoint based on validation accuracy
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name], 
        order_by=["metrics.val_acc DESC"])
    
    # Extract the top most run
    run_id = sorted_runs.loc[0,'run_id']
    
    # Get best checkpoint
    artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri).path  # get path from mlflow
    results = Result.from_path(artifact_dir)
    best_checkpoint = results.best_checkpoints[0][0]
    
    return best_checkpoint

def return_model(experiment_name):
    """Function to return the model loaded with the best checkpoint."""
    import warnings
    warnings.filterwarnings("ignore")
    
    # Get the best checkpoint
    best_checkpoint = get_best_checkpoint(experiment_name)
    
    # Initialize the model and load the checkpoints
    resnet50 = initialize_model()
    predictor = TorchPredictor.from_checkpoint(best_checkpoint, resnet50)
    return predictor

@app.command()
def predict(
        experiment_name: Annotated[str, typer.Option(help="experiment name")] = None):
    
    if experiment_name == None:
        experiment_name = "finetune-resnet-with-mlflow-tuner-1693432364"
    
    predictor = return_model(experiment_name)
    
    test_dataset = build_datasets_test()
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.numpy()
        labels = labels.numpy()
        preds = predictor.predict(inputs)
        corrects += evaluate_numpy(preds['predictions'], labels)
    
    print("Accuracy: ", corrects / len(dataloader.dataset))
    
if __name__ == "__main__":
    app()
    