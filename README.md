# Neural Network - Backpropagation Model

This project implements a neural network model using backpropagation for training on the 10 Year Diabetes Dataset. The goal is to train a model to make predictions based on the dataset and evaluate its performance. This README provides an overview of the project and its components.

## Versions

- **1.0 - Deprecated**: (Output has exceptions)
- **1.5 - Stable**: (Output is inaccurate)
- **2.0 - Previous**: (Output has exceptions)
- **2.1 - Latest**: (Output is accurate)

> ENGINEERED BY Branden van Staden.

## Overview

The challenge involved training a model on the [10 Year Diabetes Dataset](https://www.kaggle.com/datasets/jimschacko/10-years-diabetes-dataset) using Python and implementing backpropagation. The project aims to demonstrate all the steps involved in training a neural network model and provide insights into the model's performance, including logarithmic loss and accuracy.

## Project Structure

The project is organized as follows:

- `data`: Contains the dataset file (e.g., `diabetes.csv`) used for training.
- `models`: The trained models will be saved in this directory.
- `network_prototype`: Code files and resources related to the neural network prototype.

## Key Components

### Data Preprocessing

- A Pandas DataFrame is created from the dataset using the `read_csv` function.
- A profile report is generated to analyze the dataset using the `ydata_profiling` module.
- The dataset is normalized by addressing missing values, encoding features, and preprocessing it for training.

### Model Training

- The data is sliced and shuffled to create a more manageable dataset using TensorFlow's Dataset class.
- The preprocessing model is defined, which includes setting up the neural network architecture.
- The model is compiled with backpropagation (optimizer='adam').
- The model is trained using the TensorFlow `fit` function, and the trained model is saved for future use.

## Usage

To use this project, follow these steps:

1. Clone the repository from [GitHub](https://github.com/BrandenSysoutHelloWorld/myBackpropNeuralNetwork).
2. Ensure you have the necessary dependencies installed (TensorFlow, Pandas, etc.). Alternatively, you may use the `requirements.txt` file or the Dockerfile within this repo.
3. Download the dataset and place it in the `data` directory.
4. Run the project scripts to preprocess the data and train the model. Be sure to review the project documentation for detailed usage instructions.

## Challenges and Achievements

This project is a comprehensive demonstration of training a neural network model using backpropagation. Achievements include achieving a prediction accuracy of 96%, efficient data preprocessing, and model training on a dataset with 11,000 medical records. Challenges encountered during the project are outlined in the project documentation.

## Author

- Branden Van Staden

