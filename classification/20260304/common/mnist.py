import os
import gzip
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T


def get_class_names():
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def load_images(data_dir, split="train", padding=True):
    filename = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    if padding:
        data = np.pad(data, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    return data.copy()


def load_labels(data_dir, split="train"):
    filename = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.copy()


class MNISTDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, positive_class=None, padding=False):
        self.images = load_images(data_dir, split, padding=padding)
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
