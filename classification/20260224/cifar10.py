import os
import pickle
import numpy as np


TRAIN_BATCHES  = [f"data_batch_{i}" for i in range(1, 6)]
TEST_BATCHES   = ["test_batch"]


def get_class_names():
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


def _load_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    if b'data' in batch:
        batch = {k.decode('utf-8'): v for k, v in batch.items()}
    return batch


def load_images(data_dir, split="train"):
    filenames = TRAIN_BATCHES if split == "train" else TEST_BATCHES
    images_list = []
    for filename in filenames:
        batch = _load_batch(os.path.join(data_dir, "cifar-10-batches-py", filename))
        imgs = batch['data'].reshape(-1, 3, 32, 32)
        images_list.append(imgs)
    images = np.concatenate(images_list, axis=0)
    return images.transpose(0, 2, 3, 1)


def load_labels(data_dir, split="train"):
    filenames = TRAIN_BATCHES if split == "train" else TEST_BATCHES
    labels_list = []
    for filename in filenames:
        batch = _load_batch(os.path.join(data_dir, "cifar-10-batches-py", filename))
        labels_list.extend(batch['labels'])
    return np.array(labels_list, dtype='int32')


if __name__ == "__main__":

    data_dir = r"E:\datasets\cifar10"

    x_train = load_images(data_dir, "train")
    y_train = load_labels(data_dir, "train")
    x_test = load_images(data_dir, "test")
    y_test = load_labels(data_dir, "test")

    print(f"Train: {x_train.shape}, {x_train.dtype} | {y_train.shape}, {y_train.dtype}")
    print(f"Test:  {x_test.shape}, {x_test.dtype} | {y_test.shape}, {y_test.dtype}")
