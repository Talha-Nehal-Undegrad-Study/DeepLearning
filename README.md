# Deep Learning for Computer Vision

## Task 1: Image Classification with CNNs
- **Dataset and Objective:** We created a model to predict the breed of a dog or cat, given an input image. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/aseemdandgaval/23-pet-breeds-image-classification).
- **Models:** We started off with a self-made CNN-based architecture, using nn.Conv2d, nn.MaxPool2d, and nn.ReLU. After training and evaluating the model, we looked into adapting and finetuning pretrained models like InceptionNet, ResNet, EfficientNet, and MobileNet.
- **Evaluation:** We evaluated each model on the basis of accuracy, precision, recall and F1-scores.

## Task 2: Person Segmentation with Autoencoders
- **Dataset and Objective:** We created a model that outputs a segmentation mask of people inside an image. this can be used for blurring a person in an image, for example. We used the
dataset found [here](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset).
- **Models:** We started off with a simple convolution-based autoencoder and created a sequence of nn.Conv2d and nn.MaxPool2d layers that output a latent vector. We fed this into a sequence of nn.ConvTranspose2d and pooling layers that output a segmentation mask at the end.
- **Evaluation:** We plotted a few examples of our output segmentation map and compared it to the ground truth.
 we used the IoU and DICE coefficient to evalute our model.

## Task 3: Image Captioning with LSTMs
- **Dataset and Objective:** We created an LSTM model to caption images. The dataset can be found [here](https://www.kaggle.com/datasets/adityajn105/flickr8k).
- **Models:** The code explains the model used.
- **Evaluation:** We loaded our own examples to test the model.

## Task 4: Image Classification with Vision Transformers
- **Dataset and Objective:** We used the same dataset as in Task 1 with the same objective of classifying pet breeds based on an input image.
- **Models:** We used a vision transformer.
- **Evaluation:** We compared the number of parameters and the same metrics with the models from Task 1.
