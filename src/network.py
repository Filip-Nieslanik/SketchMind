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
