import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T


def get_class_names(label_type="fine"):
    if label_type == "fine":
        return [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers',
            'fruit_and_vegetables', 'household_electrical_devices',
            'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
            'large_omnivores_and_herbivores', 'medium_mammals',
            'non-insect_invertebrates', 'people', 'reptiles',
            'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
        ]
    elif label_type == "coarse":
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
            'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly',
            'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
            'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
            'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
            'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
            'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum',
            'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew',
            'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe',
            'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
    else:
        raise ValueError(f"Unknown label_type: {label_type}")


def _load_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    if b'data' in batch:
        batch = {k.decode('utf-8'): v for k, v in batch.items()}
    return batch


def load_images(data_dir, split="train"):
    filename = "train" if split == "train" else "test"
    batch = _load_batch(os.path.join(data_dir, "cifar-100-python", filename))
    images = batch['data'].reshape(-1, 3, 32, 32)
    return images.transpose(0, 2, 3, 1)


def load_labels(data_dir, split="train", label_type='fine'):
    filename = "train" if split == "train" else "test"
    batch = _load_batch(os.path.join(data_dir, "cifar-100-python", filename))
    fine = np.array(batch['fine_labels'], dtype='int32')
    coarse = np.array(batch['coarse_labels'], dtype='int32')

    if label_type == 'fine':
        return fine
    elif label_type == 'coarse':
        return coarse
    elif label_type == 'both':
        return fine, coarse
    else:
        raise ValueError(f"Unknown label_type: {label_type}")


class CIFAR100Dataset(Dataset):
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
