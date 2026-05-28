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


def test_relu_zeros_negatives():
    # relu should turn negative numbers into 0
    net = NeuralNetwork([784, 10])
    z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    out = net.relu(z)
    assert out[0] == 0.0, "relu should zero out negatives"
    assert out[1] == 0.0, "relu should zero out negatives"
    assert out[4] == 5.0, "relu should keep positives"
    print("test_relu_zeros_negatives passed")


def test_relu_deriv():
    # derivative should be 1 for positive, 0 for negative
    net = NeuralNetwork([784, 10])
    z = np.array([-2.0, 0.5, 3.0])
    d = net.relu_deriv(z)
    assert d[0] == 0.0
    assert d[1] == 1.0
    assert d[2] == 1.0
    print("test_relu_deriv passed")


def test_loss_goes_down_after_training():
    # after a few steps the loss should be lower than at the start
    net = NeuralNetwork([784, 64, 10])
    X = np.random.rand(32, 784)
    y = np.random.randint(0, 10, size=32)

    probs = net.forward(X)
    loss_before = net.loss(probs, y)

    for _ in range(50):
        net.forward(X)
        net.backward(y, lr=0.01)

    probs = net.forward(X)
    loss_after = net.loss(probs, y)

    assert loss_after < loss_before, "loss did not go down after training"
    print("test_loss_goes_down_after_training passed")


def test_weights_change_after_backward():
    # weights should be different after one backward pass
    net = NeuralNetwork([784, 64, 10])
    X = np.random.rand(8, 784)
    y = np.random.randint(0, 10, size=8)

    weights_before = [w.copy() for w in net.weights]

    net.forward(X)
    net.backward(y, lr=0.01)

    for i in range(len(net.weights)):
        assert not np.allclose(net.weights[i], weights_before[i]), f"weights[{i}] did not change"
    print("test_weights_change_after_backward passed")


def test_single_image_prediction():
    # should work with a single flat image, same as what the app sends
    net = NeuralNetwork([784, 128, 64, 10])
    img = np.random.rand(1, 784)
    digits, probs = net.predict(img)
    assert len(digits) == 1
    assert probs.shape == (1, 10)
    print("test_single_image_prediction passed")


if __name__ == "__main__":
    test_output_shape()
    test_probabilities_sum_to_one()
    test_predict_returns_valid_digit()
    test_relu_zeros_negatives()
    test_relu_deriv()
    test_loss_goes_down_after_training()
    test_weights_change_after_backward()
    test_single_image_prediction()

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        test_save_and_load(pathlib.Path(tmp))

    print("\nAll tests passed!")
