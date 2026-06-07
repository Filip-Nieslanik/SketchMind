import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        # I pass in something like [784, 128, 64, 10]
        # 784 = one value per pixel (28x28 image)
        # 128, 64 = hidden layers where the network learns patterns
        # 10 = one output per digit

        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # He initialization - I use this since plain random gave me ~11% accuracy
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        # negative values become 0, positive stay the same
        return np.maximum(0, z)

    def relu_deriv(self, z):
        # I need this in backward pass - 1 where neuron was active, 0 where it wasnt
        return (z > 0).astype(float)

    def softmax(self, z):
        # turns raw numbers into probabilities that sum to 1
        # I subtract max first to avoid overflow with large numbers
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        # pass input through each layer one by one
        # I save activations and zs here since backward pass needs them
        self.activations = [X]
        self.zs = []

        for i in range(len(self.weights)):
            z = self.activations[i] @ self.weights[i] + self.biases[i]
            self.zs.append(z)

            if i < len(self.weights) - 1:
                self.activations.append(self.relu(z))
            else:
                # last layer uses softmax to get probabilities
                self.activations.append(self.softmax(z))

        return self.activations[-1]

    def loss(self, y_pred, y_true):
        # cross-entropy loss - how wrong was the prediction
        # lower is better, 0 means perfect
        n = y_true.shape[0]
        correct = y_pred[range(n), y_true]
        return -np.log(correct + 1e-8).mean()

    def backward(self, y_true, lr):
        # go backwards through the network and figure out
        # which weights caused the error, then nudge them
        n = y_true.shape[0]
        grad = self.activations[-1].copy()
        grad[range(n), y_true] -= 1
        grad /= n

        for i in reversed(range(len(self.weights))):
            dw = self.activations[i].T @ grad
            db = grad.sum(axis=0, keepdims=True)
            if i > 0:
                # relu_deriv blocks gradient where neuron was inactive
                grad = grad @ self.weights[i].T * self.relu_deriv(self.zs[i - 1])
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs

    def save(self, path):
        # save all weights and biases into one .npz file
        data = {}
        for i in range(len(self.weights)):
            data[f"w{i}"] = self.weights[i]
            data[f"b{i}"] = self.biases[i]
        np.savez(path, **data)

    def load(self, path):
        # load weights back in the same order I saved them
        data = np.load(path)
        self.weights = []
        self.biases = []
        i = 0
        while f"w{i}" in data:
            self.weights.append(data[f"w{i}"])
            self.biases.append(data[f"b{i}"])
            i += 1
