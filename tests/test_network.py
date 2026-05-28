import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from network import NeuralNetwork


def test_output_shape():
    # network should return one probability per digit (10 total)
    net = NeuralNetwork([784, 128, 64, 10])
    X = np.random.rand(5, 784)
    probs = net.forward(X)
    assert probs.shape == (5, 10), "output shape is wrong"
    print("test_output_shape passed")


def test_probabilities_sum_to_one():
    # softmax output should always sum to 1
    net = NeuralNetwork([784, 128, 64, 10])
    X = np.random.rand(3, 784)
    probs = net.forward(X)
    sums = probs.sum(axis=1)
    assert np.allclose(sums, 1.0), "probabilities dont sum to 1"
    print("test_probabilities_sum_to_one passed")


def test_predict_returns_valid_digit():
    # predicted digit should be between 0 and 9
    net = NeuralNetwork([784, 128, 64, 10])
    X = np.random.rand(1, 784)
    digits, probs = net.predict(X)
    assert 0 <= digits[0] <= 9, "predicted digit out of range"
    print("test_predict_returns_valid_digit passed")


def test_save_and_load(tmp_path):
    # saved and loaded network should give same output
    net = NeuralNetwork([784, 128, 64, 10])
    X = np.random.rand(1, 784)
    probs_before = net.forward(X)

    path = str(tmp_path / "test_model")
    net.save(path)

    net2 = NeuralNetwork([784, 128, 64, 10])
    net2.load(path + ".npz")
    probs_after = net2.forward(X)

    assert np.allclose(probs_before, probs_after), "loaded model gives different output"
    print("test_save_and_load passed")


if __name__ == "__main__":
    test_output_shape()
    test_probabilities_sum_to_one()
    test_predict_returns_valid_digit()

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        test_save_and_load(pathlib.Path(tmp))

    print("\nAll tests passed!")
