
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from torch import nn
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Defining a function to save the model
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model = model_0,
               target_dir = "models",
               model_name = "pet_breed_classifier_V0.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents = True,
                        exist_ok = True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj = model.state_dict(),
             f = model_save_path)


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
        log_dir = os.path.join("/content/model_runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("/content/model_runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir = log_dir)

# Defining a function which returns all the metrics
def calculate_metrics(predicted_labels, ground_truth_labels):
    # Convert the tensors to numpy arrays if necessary
    predicted_labels = predicted_labels.numpy() if isinstance(predicted_labels, torch.Tensor) else predicted_labels
    ground_truth_labels = ground_truth_labels.numpy() if isinstance(ground_truth_labels, torch.Tensor) else ground_truth_labels
    
    # Calculate the metrics
    precision = precision_score(ground_truth_labels, predicted_labels, average = 'macro')
    recall = recall_score(ground_truth_labels, predicted_labels, average = 'macro')
    f1 = f1_score(ground_truth_labels, predicted_labels, average = 'macro')
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    
    return recall, precision, f1, accuracy

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
