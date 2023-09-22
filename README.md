# Melanoma_detection_Deep_learning
#Table of Contents
Introduction

Problem Statement

Objective

Data Overview

Data Source

Diseases in the Dataset

Technologies Used

Steps to Reproduce

Results and Discussion

# Introduction
Problem Statement
To build a CNN-based model that can accurately detect melanoma, a deadly form of skin cancer.

# Objective
The aim is to create a solution that can evaluate skin images and alert dermatologists about the presence of melanoma, reducing manual effort in diagnosis.

# Data Overview
Data Source

(https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166)

# Diseases in the Dataset
Actinic keratosis

Basal cell carcinoma

Dermatofibroma

Melanoma

Nevus

Pigmented benign keratosis

Seborrheic keratosis

Squamous cell carcinoma

Vascular lesion

Project Pipeline

Data Reading/Data Understanding

Defining the paths for train and test images.
 
Dataset Creation

Create train & validation datasets with a batch size of 32 and resize images to 180x180.

Dataset Visualization

Visualize one instance of all nine classes present in the dataset.

Model Building & Training

Build a CNN model that can accurately detect the 9 classes.

Use an appropriate optimizer and loss function.

Train the model for ~20 epochs.

Examine for underfitting/overfitting and apply data augmentation accordingly.

Class Distribution and Imbalance Handling

Examine class distribution in the training dataset.

Rectify class imbalances using the Augmentor library.

# Technologies Used

TensorFlow

Python

Augmentor


# Steps to Reproduce
Clone the Repository: Clone the GitHub repository to your local machine.

Install Dependencies: Install all the Python libraries and dependencies listed below down.

import pathlib

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import PIL

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

# Results and Discussion
Accuracy on training data has increased by using Augmentor library

Model is still overfitting

The problem of overfitting can be solved by add more layer,neurons or adding dropout layers.

The Model can be further improved by tuning the hyperparameter

