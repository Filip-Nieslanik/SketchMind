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
