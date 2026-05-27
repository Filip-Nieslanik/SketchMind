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


def train():
    X_train, y_train, X_test, y_test = load_dataset()

    print(f"Training images: {len(X_train)}")
    print(f"Test images:     {len(X_test)}")

    # 784 pixel values go in, probabilities for digits 0-9 come out
    net = NeuralNetwork([784, 128, 64, 10])

    epochs     = 20
    lr         = 0.01
    batch_size = 64

    print("\nStarting training...\n")

    for epoch in range(epochs):
        # shuffle training data so the network doesnt learn the order
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        total_loss = 0

        for start in range(0, len(X_train), batch_size):
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]

            y_pred = net.forward(X_batch)
            total_loss += net.loss(y_pred, y_batch)
            net.backward(y_batch, lr)

        # check how well the network does on images it has never seen
        preds, _ = net.predict(X_test)
        accuracy  = (preds == y_test).mean() * 100
        avg_loss  = total_loss / (len(X_train) / batch_size)

        print(f"Epoch {epoch + 1:2d}/{epochs}  loss: {avg_loss:.4f}  accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    train()
