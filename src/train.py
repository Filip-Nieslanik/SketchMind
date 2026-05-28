import numpy as np
import urllib.request
import gzip
import os

from network import NeuralNetwork

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

DATASET_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(DATASET_URL + filename, path)
    print("Dataset ready.")

def load_images(filename):
    path = os.path.join(DATA_DIR, filename)
    with gzip.open(path) as f:
        f.read(16)  # skip file header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(-1, 784)  # 28x28 = 784 pixels per image
    images = images / 255.0         # scale to 0-1
    return images

def load_labels(filename):
    path = os.path.join(DATA_DIR, filename)
    with gzip.open(path) as f:
        f.read(8)  # skip file header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_dataset():
    download_dataset()
    X_train = load_images(FILES["train_images"])
    y_train = load_labels(FILES["train_labels"])
    X_test  = load_images(FILES["test_images"])
    y_test  = load_labels(FILES["test_labels"])
    return X_train, y_train, X_test, y_test