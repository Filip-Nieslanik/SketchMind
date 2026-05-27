import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        # example: [784, 128, 64, 10]
        # 784 = number of pixels in one image (28x28)
        # 128, 64 = hidden layers
        # 10 = one output for each digit (0-9)
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # random starting weights, scaled down so they dont explode
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
