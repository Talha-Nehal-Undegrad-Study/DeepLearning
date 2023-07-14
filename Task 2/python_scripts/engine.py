## Defining a script for training, testing, combined train and test and evaluation

# Importing Neccessary Libraries
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple
import subprocess
import sys

# Importing torcheval for accuracy, precision and recall metrics
try:
    import torcheval
except ImportError:
    print("[INFO] Installing torcheval...")
    subprocess.check_call(['pip', 'install', 'torcheval'])

from torcheval.metrics.functional import multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score

# Import utils.py from github

try:
    import utils
except ImportError:
    print("[INFO] Cloning the repository and importing utils script...")
    subprocess.run(["git", "clone", "https://github.com/TalhaAhmed2000/DeepLearning.git"])
    subprocess.run(["mv", "DeepLearning/Task 2/python_scripts", "py_scripts"])
    sys.path.append('py_scripts')
    import utils

device = 'cuda' if torch.cuda.is_available else 'cpu'

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float, float, float, float]:

    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and Dice and IOU evaluation metrics.
    In the form (train_loss, train_dice, train_iou). For example:

    (0.1112, 0.8743, 0.0749)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_dice, train_iou = 0, 0, 0

    # Loop through data loader data batches
    for batch, (img, mask) in enumerate(dataloader):
        # Send data to target device
        img, mask = img.to(device), mask.to(device)

        # 1. Forward pass
        pred_mask = model.forward(img)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(pred_mask, mask)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate dice/iou metric across all batches
        groundtruth_mask = mask.permute(0, 2, 3, 1).cpu().numpy()
        pred_mask = pred_mask.permute(0, 2, 3, 1).cpu().detach().numpy()
        dice_coeff = round(utils.dice_coef(groundtruth_mask, pred_mask), 4)
        iou_coeff = round(utils.iou(groundtruth_mask, pred_mask), 4)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_dice = dice_coeff / len(dataloader)
    train_iou = iou_coeff / len(dataloader)
    return train_loss, train_dice, train_iou

# Testing function

# Very similar to train_step above

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing dice and iou metrics.
    In the form (test_loss, test_dice, test_iou). For eimgample:

    (0.0223, 0.8985, 0.6432)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_dice, test_iou = 0, 0, 0

    # Turn on inference conteimgt manager
    with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (img, mask) in enumerate(dataloader):
        # Send data to target device
        img, mask = img.to(device), mask.to(device)

        # 1. Forward pass
        pred_mask = model.forward(img)

        # 2. Calculate and accumulate loss
        loss = loss_fn(pred_mask, mask)
        test_loss += loss.item()

        # Calculate and accumulate accuracy

        groundtruth_mask = mask.permute(0, 2, 3, 1).cpu().numpy()
        pred_mask = pred_mask.permute(0, 2, 3, 1).cpu().detach().numpy()
        dice_coeff = round(utils.dice_coef(groundtruth_mask, pred_mask), 4)
        iou_coeff = round(utils.iou(groundtruth_mask, pred_mask), 4)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_dice = dice_coeff / len(dataloader)
    test_iou = iou_coeff / len(dataloader)
    return test_loss, test_dice, test_iou

# Overall Training Function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing dice/iou metrics. Each metric has a value in a list for
    each epoch. Also prints the time taken to train
    In the form: {train_loss: [...],
              train_dice: [...],
              train_iou: [...],
              test_loss: [...],
              test_dice: [...],
              test_iou: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_dice: [0.3945, 0.3945],
              train_iou: [0.5321, 0.4556],
              test_loss: [1.2641, 1.5706],
              test_dice: [0.3400, 0.2973],
              test_iou: [0.4321, 0.3556],
    """
    # Create empty results dictionary
    results = {"train_loss": [],
              "train_dice": [],
              "train_iou": [],
              "test_loss": [],
              "test_dice": [],
              "test_iou": []
    }


    # Make sure model on target device
    model.to(device)

    # Start Timer
    start_time = timer()

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
      train_loss, train_dice, train_iou = train_step(model, train_dataloader, loss_fn, optimizer, device)
      test_loss, test_dice, test_iou = test_step(model, test_dataloader, loss_fn, device)

      # Print out what's happening
      print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_dice: {train_dice:.4f} | "
        f"train_iou: {train_iou:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_dice: {test_dice:.4f} | "
        f"test_iou: {test_iou:.4f} | "
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_dice"].append(train_dice)
      results["train_iou"].append(train_iou)
      results["test_loss"].append(test_loss)
      results["test_dice"].append(test_dice)
      results["test_iou"].append(test_iou)

      ### New: Experiment tracking ###
      # Add loss results to SummaryWriter
      writer.add_scalars(main_tag = "Loss",
                          tag_scalar_dict = {"train_loss": train_loss,
                                          "test_loss": test_loss},
                          global_step=epoch)

      # Add dice coefficient results to SummaryWriter
      writer.add_scalars(main_tag = "Dice Coefficient",
                          tag_scalar_dict={"train_dice": train_dice,
                                          "test_dice": test_dice},
                          global_step = epoch)

      # Add iou coefficient results to SummaryWriter
      writer.add_scalars(main_tag = "IOU Coefficient",
                          tag_scalar_dict={"train_iou": train_iou,
                                          "test_iou": test_iou},
                          global_step = epoch)

      # Track the PyTorch model architecture
      writer.add_graph(model = model,
                        # Pass in an example input
                        input_to_model = torch.randn(32, 3, 256, 256).to(device))
    # close writer
    writer.close()

    # End Timer
    end_time = timer()

    utils.time_taken(start_time, end_time)

    # Return the filled results at the end of the epochs
    return results

# Evaluation/Inference Function
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device = device):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
                   
    loss, dice_coeff, iou_coeff = 0, 0, 0
    model.eval()
    with torch.inference_mode():
      for img, mask in data_loader:
        # Send data to the target device
        img, mask = img.to(device), mask.to(device)

        # Forward Pass and Loss calculation
        pred_mask = model(img)
        loss += loss_fn(pred_mask, mask)

        # Calculate and accumulate dice/iou metric across all batches
        groundtruth_mask = mask.permute(0, 2, 3, 1).cpu().numpy()
        pred_mask = pred_mask.permute(0, 2, 3, 1).cpu().detach().numpy()
        dice_coeff = utils.dice_coef(groundtruth_mask, pred_mask)
        iou_coeff = utils.iou(groundtruth_mask, pred_mask)
          
    # Adjust metrics to get average loss and accuracy per batch
    loss = loss / len(data_loader)
    dice_coeff = dice_coeff / len(data_loader)
    iou_coeff = iou_coeff / len(data_loader)

    return {"model_name": model.__class__.__name__, # onlmask works when model was created with a class
            "model_loss": loss.item(),
            "model_dice": dice_coeff,
            "model_iou": iou_coeff}
