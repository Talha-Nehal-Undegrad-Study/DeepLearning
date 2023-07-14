
# Script for task 1 and task 4 models. Given a trained models, it performs inference on random images returning their prediction, ground truth, probability with which it predicted and their plots

import torch
from torch import nn
import matplotlib.pyplot as plt
import random
import torchvision
from torchvision import transforms
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Making a function which performs inference on a single image and plots it along with its predicted and true class
def pred_and_plot_image(
    model: torch.nn.Module,
    data: torch.utils.data.dataset.Dataset,
    num_to_plot: int = 9,
    transform = None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    inception = False):
    """Makes a prediction on a set target images (randomly sampled from data) with a trained model and plots the images along with their predicted and truth label. Times the time taken for inference and takes average

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        data (dataset: torch.utils.data.dataset.Dataset): data to sample images from
        num_to_plot (int): how many images to plot. Defaults to 9
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".

    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model = model,
                            test_data,
                            num_to_plot = 4,
                            transform = torchvision.transforms.ToTensor(),
                            device = device)
    """

    # get class and their respective index as a dictionary
    class_dict = data.class_to_idx

    # Initialize test images and test labels as a list to iterate inferenc over
    test_labels = []
    test_samples = []

    # Intialize predicted labels
    pred_labels = []

    for sample, label in random.sample(list(data), k = num_to_plot):
      test_samples.append(sample)
      label = list(class_dict.keys())[label]
      test_labels.append(label)

    # Perform inference

    model.eval()
    # start timer
    start_time = timer()
    with torch.inference_mode():
      for sample in test_samples:
        # Perform inference on unsequeezed and transformed(if any) image if any
        if transform:
          sample = transform(torch.unsqueeze(sample, dim = 0)).to(device)
        sample = torch.unsqueeze(sample, dim = 0).to(device)

        # Forward Pass
        if inception:
          pred_logit, _ = model.forward(sample)
        else:
          pred_logit = model.forward(sample)
        # Get Predicted Probabilities
        pred_prob = torch.softmax(pred_logit.squeeze(), dim = 0)

        # Get Predicted class and append it along with what probability did it predict it
        idx =  torch.argmax(pred_prob).item()
        breed = list(class_dict.keys())[idx]
        breed_prob = pred_prob[idx].item()
        pred_labels.append((breed, breed_prob))

    # end timer
    end_time = timer()

    print(f'Total Time taken to perform inference on {num_to_plot} images: {end_time - start_time:.3f} seconds')
    print(f'Average Time taken to perform inference per image: {((end_time - start_time)/num_to_plot):.3f} seconds')

    # Plot the image alongside the prediction and prediction probability

    plt.figure(figsize = (16, 16))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
      # Create a subplot
      plt.subplot(nrows, ncols, i + 1)

      # Plot the target image
      plt.imshow(sample.squeeze().permute(1, 2, 0)) # Matplotlib works with

      # Find the prediction label 
      pred_label = pred_labels[i][0]

      # Find the probability with which it predicted
      pred_prob = round(pred_labels[i][1], 2)

      # Get the truth label
      truth_label = test_labels[i]

      # Create the title text of the plot
      title_text = f"Pred: {pred_label} | Truth: {truth_label} | Prob: {pred_prob}"

      # Check for equality and change title colour accordingly
      if pred_label == truth_label:
          plt.title(title_text, fontsize = 8, c="g") # green text if correct
      else:
          plt.title(title_text, fontsize = 10, c = "r") # red text if wrong
      plt.axis(False);
