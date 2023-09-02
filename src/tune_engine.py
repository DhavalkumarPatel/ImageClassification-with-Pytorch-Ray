"""
tune_engine Module

This module performs model tuning using Ray Framework.

References: 
    1. https://github.com/GokuMohandas/Made-With-ML
"""

import ray
import mlflow

from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback

# Required Imports
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler

from utils import read_config, configure_mlflow
from train_engine import train_loop_per_worker

import time

def torch_tuner() -> None:
    
    # Read the configurations from the YAML file.
    train_loop_config, scaling_config_read, mlflow_config = read_config() 
    num_runs = mlflow_config["num_runs"]
    
    # Set mlflow 
    MLFLOW_TRACKING_URI = configure_mlflow(mlflow_config)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set ray path
    storage_path = mlflow_config["ray_path"]
    
    # Configure MLflow
    experiment_name = mlflow_config["name"] + f"-{int(time.time())}"
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True)
        
    # Define the Scaling Config
    scaling_config = ScalingConfig(
        **scaling_config_read)
    
    # Define Checkpoint Config
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute ="val_acc",
        checkpoint_score_order = "max")
    
    # Define Run Config
    run_config = RunConfig(
        name=experiment_name,
        storage_path=storage_path,
        callbacks=[mlflow_callback], 
        checkpoint_config=checkpoint_config,
    )
    
    # Torch Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config
    )
    
    # Set the Scheduler
    scheduler = ASHAScheduler(
        max_t=train_loop_config["num_epochs"],  # max epoch (<time_attr>) per trial
        grace_period=1,  # min epoch (<time_attr>) per trial
    )
    
    # Set the parameter space
    param_space = {
        "train_loop_config": {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128, 256])
            }
        }
    
    # Tune config
    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode="min",
        scheduler=scheduler,
        num_samples=num_runs,
    )
    
    # Tuner
    tuner = Tuner(
        trainable=trainer,
        run_config=run_config,
        param_space=param_space,
        tune_config=tune_config,
    )
    
    # Fit
    results = tuner.fit()
    best_trial = results.get_best_result(metric="val_loss", mode="min")
    print(best_trial)
    
if __name__ == "__main__": 
    if ray.is_initialized():
        ray.shutdown()
    torch_tuner()
    
    
    
    
    
    
    
    