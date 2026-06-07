import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        # layer_sizes tells me how big each layer is
        # I pass in [784, 128, 64, 10] from app.py and train.py
        # 784 = one value per pixel (28x28 image flattened)
        # 128, 64 = hidden layers, this is where the network learns patterns
        # 10 = one output per digit (0-9)
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases  = []

        for i in range(len(layer_sizes) - 1):
            # He initialization - I multiply by sqrt(2/n) since plain random weights
            # gave me only ~11% accuracy, this made it converge properly
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        # negative values become 0, positive stay the same
        # I use this after each hidden layer so inactive neurons don't affect the result
        return np.maximum(0, z)

    def relu_deriv(self, z):
        # I need this in backward() to know which neurons were active
        # active neuron (z > 0) = gradient passes through
        # inactive neuron (z <= 0) = gradient is blocked, weight stays unchanged
        return (z > 0).astype(float)

    def softmax(self, z):
        # converts 10 raw numbers into probabilities that sum to 1
        # I use this only on the last layer so I can read the output as percentages
        # subtracting max prevents overflow when numbers get very large
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        # runs the input through all layers one by one
        # called from predict() in app.py every time I draw a stroke
        # also called from train.py during training to get predictions

        # I store activations and zs because backward() needs them later
        self.activations = [X]
        self.zs = []

        for i in range(len(self.weights)):
            # z = weighted sum of previous layer + bias
            z = self.activations[i] @ self.weights[i] + self.biases[i]
            self.zs.append(z)

            if i < len(self.weights) - 1:
                # hidden layers use ReLU
                self.activations.append(self.relu(z))
            else:
                # last layer uses softmax to get probabilities for digits 0-9
                self.activations.append(self.softmax(z))

        return self.activations[-1]

    def loss(self, y_pred, y_true):
        # measures how wrong the prediction was - lower is better
        # I only look at the probability the network gave to the correct digit
        # if it said 95% for the right answer, loss is small
        # if it said 3% for the right answer, loss is big
        # called from train.py after every batch to track progress
        n = y_true.shape[0]
        correct = y_pred[range(n), y_true]
        return -np.log(correct + 1e-8).mean()  # 1e-8 avoids log(0)

    def backward(self, y_true, lr):
        # figures out which weights caused the error and nudges them in the right direction
        # called right after forward() in train.py during every batch
        # lr = learning rate, controls how big the nudge is (I use 0.01)
        n = y_true.shape[0]

        # start gradient from the output layer
        # subtract 1 from the correct class - this is the derivative of softmax + cross-entropy
        grad = self.activations[-1].copy()
        grad[range(n), y_true] -= 1
        grad /= n

        # walk backwards through layers
        for i in reversed(range(len(self.weights))):
            dw = self.activations[i].T @ grad   # gradient for weights in this layer
            db = grad.sum(axis=0, keepdims=True) # gradient for biases in this layer

            if i > 0:
                # pass gradient to previous layer
                # relu_deriv blocks it where the neuron was inactive
                grad = grad @ self.weights[i].T * self.relu_deriv(self.zs[i - 1])

            # update weights and biases
            self.weights[i] -= lr * dw
            self.biases[i]  -= lr * db

    def predict(self, X):
        # runs forward pass and returns the winning digit + all probabilities
        # called from run_prediction() in app.py on every draw stroke
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs

    def save(self, path):
        # saves all weights and biases into a single .npz file
        # called at the end of train.py, saved to model/model.npz
        data = {}
        for i in range(len(self.weights)):
            data[f"w{i}"] = self.weights[i]
            data[f"b{i}"] = self.biases[i]
        np.savez(path, **data)

    def load(self, path):
        # loads weights back in the same order they were saved
        # called in App.__init__() in app.py before the window opens
        data = np.load(path)
        self.weights = []
        self.biases  = []
        i = 0
        while f"w{i}" in data:
            self.weights.append(data[f"w{i}"])
            self.biases.append(data[f"b{i}"])
            i += 1