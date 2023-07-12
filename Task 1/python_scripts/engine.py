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
    subprocess.run(["mv", "DeepLearning/Task 1/python_scripts", "py_scripts"])
    sys.path.append('py_scripts')
    import utils

device = 'cuda' if torch.cuda.is_available else 'cpu'

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               num_classes: int,
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
    num_classes: Number of classes (Binary or Multi)
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training evaluation metrics.
    In the form (train_loss, train_accuracy, train_precision, train_recall, train_f1). For example:

    (0.1112, 0.8743, 0.7813, 0.7945, 0.8441)
    """

    # Put Model in train_mode
    model.train()

    # Initialize train loss and other metrics as 0. Per epoch, we will be looking at each of these per batch
    train_loss, train_accuracy, train_precision, train_recall, train_f1 = 0, 0, 0, 0, 0

    # Loop through the data loader
    for batch, (X, y) in enumerate(dataloader):

      # Respective Device Transition
      X, y = X.to(device), y.to(device)

      # Forward Pass (In case of Inception net, we get output and something else but we dont need that as the loss function would give an error otherwise)
      y_logits, _ = model.forward(X)

      # Calculate loss (acumulative)
      loss = loss_fn(y_logits, y)
      train_loss += loss.item()

      # Reset Optimizer
      optimizer.zero_grad()

      # BackPropogation
      loss.backward()

      # Optimizer Step
      optimizer.step()

      # Calculate the predicted class using softmax since multi-class and then taking argmax (on the dim = 1 since 0 corresponds to batch)
      y_pred_class = torch.argmax(torch.softmax(y_logits, dim = 1), dim = 1)

      # Get num_classes as the number of unique elements in y
      num_classes = len(torch.unique(y))  
      # Calculate Evaluation Metrics
      train_accuracy += multiclass_accuracy(y_pred_class, y)
      train_precision += multiclass_precision(y_pred_class, y, average = 'macro', num_classes = num_classes)
      train_recall += multiclass_recall(y_pred_class, y, average = 'micro', num_classes = num_classes)
      train_f1 += multiclass_f1_score(train_precision, train_recall, average = 'macro', num_classes = num_classes)

    # Calculate each loss and each metric per batch
    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)
    train_precision = train_precision / len(dataloader)
    train_recall = train_recall / len(dataloader)
    train_f1 = train_f1 / len(dataloader)

    return train_loss, train_accuracy, train_precision, train_recall, train_f1


# Testing function

# Very similar to train_step above

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               num_classes: int,
               device: torch.device) -> Tuple[float, float, float, float, float]:

    """tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to testing mode and then
    runs through all of the required testing steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    num_classes: Number of classes (Binary or Multi)
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing evaluation metrics.
    In the form (test_loss, test_accuracy, test_precision, test_recall, test_f1). For example:

    (0.1112, 0.8743, 0.7813, 0.7945, 0.8441)
    """

    # Put Model in test_mode
    model.eval()

    # Initialize test loss and other metrics as 0. Per epoch, we will be looking at each of these per batch
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0

    # Loop through the data loader
    with torch.inference_mode():
      for batch, (X, y) in enumerate(dataloader):

        # Respective Device Transition
        X, y = X.to(device), y.to(device)

        # Forward Pass (In case of Inception net, we get output and something else but we dont need that as the loss function would give an error otherwise)
        y_logits, _ = model.forward(X)

        # Calculate loss (acumulative)
        loss = loss_fn(y_logits, y)
        test_loss += loss.item()

        # Get num_classes as the number of unique elements in y
        num_classes = len(torch.unique(y))
          
        # Calculate the predicted class using softmax since multi-class and then taking argmax (on the dim = 1 since 0 corresponds to batch)
        y_pred_class = torch.argmax(torch.softmax(y_logits, dim = 1), dim = 1)
        test_accuracy += multiclass_accuracy(y_pred_class, y)
        test_precision += multiclass_precision(y_pred_class, y, average = 'macro', num_classes = num_classes)
        test_recall += multiclass_recall(y_pred_class, y, average = 'micro', num_classes = num_classes)
        test_f1 += multiclass_f1_score(test_precision, test_recall, average = 'macro', num_classes = num_classes)

    # Calculate each loss and each metric per batch
    test_loss = test_loss / len(dataloader)
    test_accuracy = test_accuracy / len(dataloader)
    test_precision = test_precision / len(dataloader)
    test_recall = test_recall / len(dataloader)
    test_f1 = test_f1 / len(dataloader)

    return test_loss, test_accuracy, test_precision, test_recall, test_f1

# Overall Training Function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          num_classes: int,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
              
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout as well as adds results to TensorBoard

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
    testing accuracy metrics. Each metric has a value in a list for
    each epoch. Also prints the time taken to train
    In the form: {train_loss: [...],
              train_acc: [...],
              train_precision: [...],
              train_recall: [...],
              train_f1: [...],
              test_loss: [...],
              test_acc: [...],
              test_precision: [...],
              test_recall: [...],
              test_f1: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              train_precision: [0.5321, 0.4556],
              train_recall: [0.4521, 0.4241],
              train_f1: [0.3402, 0.4671],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973],
              test_precision: [0.4321, 0.3556],
              test_recall: [0.3521, 0.3241],
              test_f1: [0.2402, 0.3671]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
              "train_acc": [],
              "train_precision": [],
              "train_recall": [],
              "train_f1": [],
              "test_loss": [],
              "test_acc": [],
              "test_precision": [],
              "test_recall": [],
              "test_f1": []
    }


    # Make sure model on target device
    model.to(device)

    # Start Timer
    start_time = timer()

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
      train_loss, train_acc, train_precision, train_recall, train_f1 = train_step(model, train_dataloader, loss_fn, optimizer, num_classes, device)
      test_loss, test_acc, test_precision, test_recall, test_f1 = test_step(model, test_dataloader, loss_fn, num_classes, device)

      # Print out what's happening
      print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"train_precision: {train_precision:.4f} | "
        f"train_recall: {train_recall:.4f} | "
        f"train_f1: {train_f1:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f} | "
        f"test_acc: {test_acc:.4f} | "
        f"test_precision: {test_precision:.4f} | "
        f"test_recall: {test_recall:.4f} | "
        f"test_f1: {test_f1:.4f} | "
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["train_precision"].append(train_precision)
      results["train_recall"].append(train_recall)
      results["train_f1"].append(train_f1)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
      results["test_precision"].append(test_precision)
      results["test_recall"].append(test_recall)
      results["test_f1"].append(test_f1)

      ### Experiment tracking on Tensorboard ###
      # Add loss results to SummaryWriter
      writer.add_scalars(main_tag = "Loss",
                          tag_scalar_dict = {"train_loss": train_loss,
                                          "test_loss": test_loss},
                          global_step = epoch)

      # Add accuracy results to SummaryWriter
      writer.add_scalars(main_tag = "Accuracy",
                          tag_scalar_dict={"train_acc": train_acc,
                                          "test_acc": test_acc},
                          global_step = epoch)

      # Add precision results to SummaryWriter
      writer.add_scalars(main_tag = "Precision",
                          tag_scalar_dict={"train_precision": train_precision,
                                          "test_precision": test_precision},
                          global_step = epoch)

      # Add Recall results to SummaryWriter
      writer.add_scalars(main_tag = "Recall",
                          tag_scalar_dict={"train_recall": train_recall,
                                          "test_recall": test_recall},
                          global_step = epoch)


      # Add f1 results to SummaryWriter
      writer.add_scalars(main_tag = "F1 Score",
                          tag_scalar_dict={"train_f1": train_f1,
                                          "test_f1": test_f1},
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
               num_classes: int,
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
    loss, accuracy, precision, recall, f1 = 0, 0, 0, 0, 0
    model.eval()
    with torch.inference_mode():
      for X, y in data_loader:
        # Send data to the target device
        X, y = X.to(device), y.to(device)
        y_pred, _ = model(X)
        loss += loss_fn(y_pred, y)
          
        # Get num_classes as the number of unique elements in y
        num_classes = len(torch.unique(y))
          
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
        accuracy += multiclass_accuracy(y_pred_class, y)
        precision += multiclass_precision(y_pred_class, y, average = 'macro', num_classes = num_classes)
        recall += multiclass_recall(y_pred_class, y, average = 'micro', num_classes = num_classes)
        f1 += multiclass_f1_score(precision, recall, average = 'macro', num_classes = num_classes)

      # Scale loss and acc
      loss /= len(data_loader)
      accuracy /= len(data_loader)
      precision /= len(data_loader)
      recall /= len(data_loader)
      f1 /= len(data_loader)
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_accuracy": accuracy,
            "model_precision": precision,
            "model_recall": recall,
            "model_f1": f1}
