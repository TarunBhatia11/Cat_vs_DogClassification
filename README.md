# Cat and Dog Classification

This repository contains a machine learning model for classifying images of cats and dogs using Convolutional Neural Networks (CNN) and Deep Neural Networks (DNN) with Python.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)


## Introduction

This project aims to build a machine learning model that can accurately classify images of cats and dogs. The model is implemented using CNN and DNN techniques in Python. The model is trained on a dataset of labeled images of cats and dogs and evaluated to measure its performance.

## Dataset

The dataset used for training and evaluating the model is the [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765). The dataset contains a total of 25,000 images of cats and dogs, with 12,500 images of each class.

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/TarunBhatia11/cat-and-dog-classification.git
cd cat-and-dog-classification
pip install -r requirements.txt
```

## Usage

To train and evaluate the model, run the following command:

```bash
python train.py
```

To make predictions on new images, use the following command:

```bash
python predict.py --image_path path_to_image.jpg
```

## Model Architecture

The model is built using a Convolutional Neural Network (CNN) architecture followed by a Deep Neural Network (DNN) for classification. The architecture consists of the following layers:

1. Convolutional Layers
2. Max Pooling Layers
3. Fully Connected Layers
4. Dropout Layers
5. Output Layer

## Training

The model is trained using the Adam optimizer and categorical cross-entropy loss. The training script `train.py` handles the data preprocessing, model training, and saving the trained model. The training parameters such as learning rate, batch size, and number of epochs can be configured in the `config.json` file.

## Evaluation

The model's performance is evaluated using accuracy, precision, recall, and F1-score metrics. The evaluation script `evaluate.py` loads the trained model and evaluates it on the test dataset.

## Results

The model achieves an accuracy of XX% on the test dataset. Below are some sample results:

| Metric     | Value |
|------------|-------|
| Accuracy   | XX%   |
| Precision  | XX%   |
| Recall     | XX%   |
| F1-Score   | XX%   |

## Disclaimer

This repository is generally modelized for educational purpose.

