import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        # layer_sizes is a list like [784, 128, 64, 10]
        # 784 = pixels in one image (28x28)
        # 128 and 64 = hidden layers
        # 10 = one output per digit (0-9)

        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        # if negative set to 0
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        # convert numbers to probabilities, all sum to 1
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        # go through each layer one by one
        # save everything so backward pass can use it later
        self.activations = [X]
        self.zs = []

        for i in range(len(self.weights)):
            z = self.activations[i] @ self.weights[i] + self.biases[i]
            self.zs.append(z)

            if i < len(self.weights) - 1:
                self.activations.append(self.relu(z))
            else:
                self.activations.append(self.softmax(z))

        return self.activations[-1]

    def loss(self, y_pred, y_true):
        # how wrong was the prediction, lower is better
        n = y_true.shape[0]
        correct = y_pred[range(n), y_true]
        return -np.log(correct + 1e-8).mean()

    def backward(self, y_true, lr):
        # figure out how much each weight was responsible for the error
        # then nudge it in the right direction
        n = y_true.shape[0]
        grad = self.activations[-1].copy()
        grad[range(n), y_true] -= 1
        grad /= n

        for i in reversed(range(len(self.weights))):
            dw = self.activations[i].T @ grad
            db = grad.sum(axis=0, keepdims=True)
            if i > 0:
                grad = grad @ self.weights[i].T * self.relu_deriv(self.zs[i - 1])
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs

    def save(self, path):
        data = {}
        for i in range(len(self.weights)):
            data[f"w{i}"] = self.weights[i]
            data[f"b{i}"] = self.biases[i]
        np.savez(path, **data)

    def load(self, path):
        data = np.load(path)
        self.weights = []
        self.biases = []
        i = 0
        while f"w{i}" in data:
            self.weights.append(data[f"w{i}"])
            self.biases.append(data[f"b{i}"])
            i += 1
