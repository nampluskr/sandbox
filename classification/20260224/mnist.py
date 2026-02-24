import os
import gzip
import numpy as np


TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"


def get_class_names(mnist_type="mnist"):
    if mnist_type == "mnist":
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif mnist_type == "fashion":
        return [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    else:
        raise ValueError(f"Unknown mnist_type: {mnist_type}")


def load_images(data_dir, split="train"):
    filename = TRAIN_IMAGES if split == "train" else TEST_IMAGES
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_labels(data_dir, split="train"):
    filename = TRAIN_LABELS if split == "train" else TEST_LABELS
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


if __name__ == "__main__":

    data_dir = r"E:\datasets\mnist"
    
    x_train = load_images(data_dir, "train")
    y_train = load_labels(data_dir, "train")
    x_test = load_images(data_dir, "test")
    y_test = load_labels(data_dir, "test")
    
    print(f"Train: {x_train.shape}, {x_train.dtype} | {y_train.shape}, {y_train.dtype}")
    print(f"Test:  {x_test.shape}, {x_test.dtype} | {y_test.shape}, {y_test.dtype}")
