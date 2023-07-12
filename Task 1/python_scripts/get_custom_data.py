
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def getImageFolderDataLoaders(train_dir: str,
                         test_dir: str,
                         transform: torchvision.transforms.Compose,
                         batch: int = 32):
  """
  Transforms a dataset into a custom Datset subclassed by ImageFolder and into dataloaders for both test and train
  Assumes the file structure of the data to be in the following

  data/ <- overall dataset folder
    train/ <- training images
        siamese cat/ <- class name as folder name
            image01.jpg
            image02.jpg
            ...
        ragdoll cat/
            image24.jpg
            image25.jpg
            ...
        sphinx/
            image37.jpg
            ...
    test/ <- testing images
        siamese cat/ <- class name as folder name
            image101.jpg
            image102.jpg
            ...
        ragdoll cat/
            image124.jpg
            image125.jpg
            ...
        sphinx/
            image137.jpg
            ...

  Args:
  train_dir (str): Directory where the training images are stored
  test_dir (str): Directory where the test images are stored
  transform (trochvision.transforms.Compose): A set of transforms to apply on the images
  batch (int): Batch size for the dataloaders. Defaults to 32

  Returns: 
  train_data --> subclass of torch.utils.data.Dataset
  test_data --> subclass of torch.utils.data.Dataset
  train_dataloader --> an iterable object of torch.utils.data.DataLoader
  test_dataloader --> an iterable object of torch.utils.data.DataLoader
  """
  
  # Get Custom Dataset
  train_data = datasets.ImageFolder(root = train_dir,
                                    transform = transform) 
                                    

  test_data = datasets.ImageFolder(root = test_dir,
                                  transform = transform)

  # Get Dataloaders  
  train_dataloader = DataLoader(dataset = train_data,
                              batch_size = batch,
                              num_workers = 1, 
                              shuffle = True) 

  test_dataloader = DataLoader(dataset = test_data,
                              batch_size = batch,
                              num_workers = 1,
                              shuffle = False) 

  return train_data, test_data, train_dataloader, test_dataloader
