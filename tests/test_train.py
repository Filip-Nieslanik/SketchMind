import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from train import load_images, load_labels, FILES


def test_image_shape():
    # each image should be 784 pixels (28x28 flattened)
    images = load_images(FILES["train_images"])
    assert images.shape[1] == 784, "images should have 784 pixels"
    print("test_image_shape passed")


def test_image_count():
    # MNIST training set has exactly 60000 images
    images = load_images(FILES["train_images"])
    assert images.shape[0] == 60000, "expected 60000 training images"
    print("test_image_count passed")


def test_pixel_range():
    # pixels should be normalized between 0 and 1
    images = load_images(FILES["train_images"])
    assert images.min() >= 0.0, "pixels below 0"
    assert images.max() <= 1.0, "pixels above 1"
    print("test_pixel_range passed")


def test_label_count():
    # should have one label per image
    images = load_images(FILES["train_images"])
    labels = load_labels(FILES["train_labels"])
    assert len(labels) == len(images), "label count doesnt match image count"
    print("test_label_count passed")


def test_label_range():
    # labels should only be digits 0 through 9
    labels = load_labels(FILES["train_labels"])
    assert labels.min() >= 0, "label below 0"
    assert labels.max() <= 9, "label above 9"
    print("test_label_range passed")


def test_test_set_size():
    # MNIST test set has exactly 10000 images
    images = load_images(FILES["test_images"])
    assert images.shape[0] == 10000, "expected 10000 test images"
    print("test_test_set_size passed")


if __name__ == "__main__":
    test_image_shape()
    test_image_count()
    test_pixel_range()
    test_label_count()
    test_label_range()
    test_test_set_size()

    print("\nAll tests passed!")
