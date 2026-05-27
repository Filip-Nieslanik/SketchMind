import numpy as np
import os
from tensorflow import keras

from network import NeuralNetwork


DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def load_dataset():
    # keras downloads the dataset automatically on first run
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # flatten each 28x28 image into 784 numbers and scale to 0-1
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test  = X_test.reshape(-1, 784)  / 255.0

    return X_train, y_train, X_test, y_test
