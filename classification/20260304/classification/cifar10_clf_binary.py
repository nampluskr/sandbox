import sys

ROOT_DIR = "/home/namu/myspace/NAMU/tutorials"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common.utils import set_seed
from common.cifar10 import CIFAR10Dataset
from common.trainer import fit, evaluate
from models.classifier import CNN, BinaryClassifier


if __name__ == "__main__":
    print(f">> {os.path.basename(__file__)}")

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = "/home/namu/myspace/NAMU/datasets/cifar10"
    NUM_CLASSES = 1
    SEED = 42
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10

    set_seed(SEED)

    #################################################################
    # Data loading
    #################################################################
    train_dataset = CIFAR10Dataset(DATA_DIR, "train", positive_class=0)
    test_dataset = CIFAR10Dataset(DATA_DIR, "test", positive_class=0)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

    print(f"\n>> Train Data (Batch):")
    train_batch = next(iter(train_loader))
    x, y = train_batch["image"], train_batch["label"]
    print(f"train images: {x.shape}, {x.dtype}")
    print(f"train labels: {y.shape}, {y.dtype}")

    #################################################################
    # Modeling
    #################################################################
    model = CNN(num_classes=NUM_CLASSES, in_channels=3)
    clf = BinaryClassifier(model)       # optimizer, loss_fn, accuracy, device

    print(f"\n>> Training:")
    history = fit(clf, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)

    print(f"\n>> Evaluation:")
    test_results = evaluate(clf, test_loader)
    print(", ".join([f"{k}={v:.3f}" for k, v in test_results.items()]))

    print(f"\n>> Prediction:")
    test_batch = next(iter(test_loader))
    images = test_batch["image"][:NUM_SAMPLES]
    labels = test_batch["label"][:NUM_SAMPLES]

    preds = clf.predict(images)
    pred_labels = (preds >= 0.5).int()
    for i in range(NUM_SAMPLES):
        print(f"Target: {labels[i].item()} | Prediction: {pred_labels[i].item()} (prob: {preds[i].item():.3f})")

