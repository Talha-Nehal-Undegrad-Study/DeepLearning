
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str,
                 split: str = 'train',
                 transform: torchvision.transforms.Compose = None):
      """ Assumes the structure of the data is in the standard image data format

      Args:
        root_dir (str): Target directory to access the images and the masks from
        split (str): Which split to perform; train or test? Defaults to train
        transform (torchvision.transforms.Compose): Which data augmentation to apply on the images

      Methods:
        len(): Returns the length of the dataset
        getitem(index): Returns the image and its corresponding mask corresponding to the index in the data
      """

      # Initialize all directories

      self.root_dir = root_dir
      self.transform = transform

      self.image_dir = os.path.join(root_dir, 'images')
      self.mask_dir = os.path.join(root_dir, 'mask')

      # 1) Initialize train/test directories of images/masks which will be updated according to the split
      # 2) Get hold of image and mask files

      if split == 'train':
        self.image_train_test_dir = os.path.join(self.image_dir, 'train')
                                                                                # 1)
        self.mask_train_test_dir = os.path.join(self.mask_dir, 'train')

        self.image_files = os.listdir(os.path.join(self.image_dir, 'train'))
                                                                                # 2)
        self.mask_files = os.listdir(os.path.join(self.mask_dir, 'train'))

      elif split == 'test':
        self.image_train_test_dir = os.path.join(self.image_dir, 'test')
                                                                                # 1)
        self.mask_train_test_dir = os.path.join(self.mask_dir, 'test')

        self.image_files = os.listdir(os.path.join(self.image_dir, 'test'))
                                                                                # 2)
        self.mask_files = os.listdir(os.path.join(self.mask_dir, 'test'))

    # Overrides len function
    def __len__(self):
      return len(self.image_files)

    # Overrides getitem function
    def __getitem__(self, index):

      # Get the path to the image and mask corresponding to the index
      image_path = os.path.join(self.image_train_test_dir, self.image_files[index])
      mask_path = os.path.join(self.mask_train_test_dir, self.mask_files[index])

      image = Image.open(image_path)
      mask = Image.open(mask_path)

      # Perform data augmentation if required
      if self.transform is not None:
          image = self.transform(image)
          mask = self.transform(mask)

      # Return image and corresponding Mask
      return image, mask


