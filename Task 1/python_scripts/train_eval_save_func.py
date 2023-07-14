
# Script for doing all of training process, eval_process, and save model

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
import os

# Get relevant scripts from GitHub
try:
    import utils, engine
except ImportError:
    print("[INFO] Cloning the repository and importing utils script...")
    subprocess.run(["git", "clone", "https://github.com/TalhaAhmed2000/DeepLearning.git"])
    subprocess.run(["mv", "DeepLearning/Task 1/python_scripts", "py_scripts"])
    sys.path.append('py_scripts')
    import utils, engine

def train_eval_save(model: torch.nn.Module,
                    train_dataloader: torch.utils.data.DataLoader
                    test_dataloader: torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epochs: int,
                    num_classes: int,
                    device: torch.device,
                    writer: torch.utils.tensorboard.writer.SummaryWriter,
                    target_dir: str,
                    model_name: str,
                    inception = False)

  """"
  Receives a model --> trains the model --> evaluates the model --> saves the model (specifically its parameters only)

  Args:
  model (torch.nn.Module): Model to train, evaluate and save
  train_dataloader (torch.utils.data.DataLoader): Data for the model to be trained on
  test_dataloader (torch.utils.data.DataLoader): Data for the model to be tested on
  loss_fn (torch.nn.Module): The loss function which the model aims to minimize during training phase
  optimizer (torch.optim.Optimizer): The optimizer model uses to update the weights/parameters
  epochs (int): How many epochs the model should train
  device (torch.device): Target device to do computation ('cuda' or 'cpu')
  writer (torch.utils.tensorboard.writer.SummaryWriter): Instance of tensorboard writer to get logs of the result evaluation metrics e.g loss, accuracy
  target_dir (str): The directory in which the model should be saved
  model_name (str): With which name the model it should be saved

  Returns: 
  Evaluation results gotten from the engine.eval() function, i.e. a dictionary containing the summary of the model with its results
  """

  # Train the model
  engine.train(model = model,
              train_dataloader = train_dataloader,
              test_dataloader = test_dataloader,
              optimizer = optimizer,
              loss_fn = loss_fn,
              num_classes = len(class_names),
              epochs = epochs,
              device = device,
              writer = writer,
              inception = inception)

  # Evaluate the model and store results
  model_results = engine.eval(model = model,
                              dataloader = test_dataloader,
                              loss_fn = loss_fn,
                              num_classes = num_classes,
                              device = device,
                             inception = inception)

  # Save Model
  utils.save_model(model = model,
                  target_dir = target_dir,
                  model_name = model_name)
  
  return model_results
