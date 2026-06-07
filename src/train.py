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
        f.read(16)  # first 16 bytes are header info I don't need
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(-1, 784)  # each image is 28x28 = 784 pixels
    images = images / 255.0         # I scale to 0-1 since the network works better that way
    return images

def load_labels(filename):
    path = os.path.join(DATA_DIR, filename)
    with gzip.open(path) as f:
        f.read(8)  # skip header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_dataset():
    download_dataset()
    X_train = load_images(FILES["train_images"])
    y_train = load_labels(FILES["train_labels"])
    X_test  = load_images(FILES["test_images"])
    y_test  = load_labels(FILES["test_labels"])
    return X_train, y_train, X_test, y_test


def train():
    X_train, y_train, X_test, y_test = load_dataset()

    print(f"Training images: {len(X_train)}")
    print(f"Test images:     {len(X_test)}")

    net = NeuralNetwork([784, 128, 64, 10])

    epochs     = 20
    lr         = 0.01
    batch_size = 64

    print("\nStarting training...\n")

    for epoch in range(epochs):
        # shuffle every epoch so the network doesn't just memorize the order
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        total_loss = 0

        for start in range(0, len(X_train), batch_size):
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]

            y_pred      = net.forward(X_batch)
            total_loss += net.loss(y_pred, y_batch)
            net.backward(y_batch, lr)

        preds, _  = net.predict(X_test)
        accuracy  = (preds == y_test).mean() * 100
        avg_loss  = total_loss / (len(X_train) / batch_size)

        print(f"Epoch {epoch + 1:2d}/{epochs}  loss: {avg_loss:.4f}  accuracy: {accuracy:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model")
    net.save(model_path)
    print(f"\nModel saved to {model_path}.npz")


if __name__ == "__main__":
    train()
