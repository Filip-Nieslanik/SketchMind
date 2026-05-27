import numpy as np
import urllib.request
import gzip
import os

from network import NeuralNetwork


DATASET_URL = "http://yann.lecun.com/exdb/mnist/"
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def download_dataset():
    # download only files that are not already on disk
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
        f.read(16)  # first 16 bytes are file metadata
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(-1, 784)  # each image is 28x28 = 784 pixels
    images = images / 255.0         # scale pixel values from 0-255 to 0-1
    return images


def load_labels(filename):
    path = os.path.join(DATA_DIR, filename)
    with gzip.open(path) as f:
        f.read(8)  # first 8 bytes are file metadata
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels
