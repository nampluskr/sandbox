import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T


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


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, positive_class=None):
        self.images = load_images(data_dir, split)
        self.labels = load_labels(data_dir, split)
        self.transform = transform or T.ToTensor()
        self.positive_class = positive_class

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        if self.positive_class is not None:
            label = (label == self.positive_class)
        label = torch.tensor(label).long()
        return {"image": image, "label": label}
