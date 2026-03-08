import sys

ROOT_DIR = "/home/namu/myspace/NAMU/tutorials/oxford_pets_tutorials"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

from common.utils import set_seed
from common.oxford_pets import OxfordPetsClassification, collate_fn
from common.train import fit, evaluate
from common.classifier import CNNModel, Classifier


if __name__ == "__main__":
    print(f">> {os.path.basename(__file__)}")

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = "/home/namu/myspace/NAMU/datasets/oxford_pets"
    NUM_CLASSES = 37
    SEED = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10

    set_seed(SEED)

    #################################################################
    # Data loading
    #################################################################
    train_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = OxfordPetsClassification(DATA_DIR, "train", transform=train_transform)
    test_dataset = OxfordPetsClassification(DATA_DIR, "test", transform=test_transform)

    train_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": collate_fn,
        "pin_memory": True,
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    test_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": True,
        "num_workers": 4,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    print(f"\n>> Train Data (Batch):")
    train_batch = next(iter(train_loader))
    x, y = train_batch["image"], train_batch["label"]
    print(f"train images: {x.shape}, {x.dtype}, [{x.min():.2f}, {x.max():.2f}]")
    print(f"train labels: {y.shape}, {y.dtype}, [{y.min()}, {y.max()}]")

    #################################################################
    # Modeling
    #################################################################
    model = CNNModel(num_classes=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    clf = Classifier(model, optimizer=optimizer, num_classes=NUM_CLASSES)    # optimizer, loss_fn, accuracy, device

    print(f"\n>> Training:")
    history = fit(clf, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)
    # history = fit(clf, train_loader, num_epochs=NUM_EPOCHS)

    print(f"\n>> Evaluation:")
    test_results = evaluate(clf, test_loader)
    print(", ".join([f"{k}={v:.3f}" for k, v in test_results.items()]))

    print(f"\n>> Prediction:")
    test_batch = next(iter(test_loader))
    images = test_batch["image"][:NUM_SAMPLES]
    labels = test_batch["label"][:NUM_SAMPLES]

    preds = clf.predict(images)
    for i in range(NUM_SAMPLES):
        print(f"Target: {labels[i].item()} | Prediction: {preds[i].argmax()}")

    del train_loader
    del test_loader
