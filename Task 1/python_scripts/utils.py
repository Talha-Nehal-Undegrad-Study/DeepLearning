
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str = None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates tensorboard and saves it to a specified log directory.

    log directory is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/pet_breed_80_20_split/AnimalBreedClassifierV0/5_epochs/"
        writer = create_writer(experiment_name = "pet_breed_80_20_split",
                               model_name = "AnimalBreedClassifierV0",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date 
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        # Create log directory path
        log_dir = os.path.join("content/model_runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("content/model_runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir = log_dir)


# Defining a function to print the time taken to train a model
def time_taken(start, end):
  print(f'\nTrain Time: {end -start:.3f} seconds')

# Defining functions for IOU and Dice coefficient metrics to evaluate the quality of mask segmentation in Task 2

def iou(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    iou = np.mean(intersect/union)
    return round(iou, 3)

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3) #round up to 3 decimal places
