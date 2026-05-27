import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        # layer_sizes tells us how big each layer is
        # example: [784, 128, 64, 10]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        # negative values become 0, positive stay the same
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        # turn output numbers into probabilities (they all add up to 1)
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        # pass the input through every layer one by one
        self.activations = [X]
        self.zs = []

        for i in range(len(self.weights)):
            z = self.activations[i] @ self.weights[i] + self.biases[i]
            self.zs.append(z)

            # hidden layers use relu, last layer uses softmax
            if i < len(self.weights) - 1:
                self.activations.append(self.relu(z))
            else:
                self.activations.append(self.softmax(z))

        return self.activations[-1]

    def loss(self, y_pred, y_true):
        # measures how wrong the prediction is, lower is better
        n = y_true.shape[0]
        correct_probs = y_pred[range(n), y_true]
        return -np.log(correct_probs + 1e-8).mean()

    def backward(self, y_true, lr):
        # figure out how wrong each weight was and fix it a little bit
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
        # run forward pass and pick the digit with highest probability
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs
