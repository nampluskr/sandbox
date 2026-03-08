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
from common.oxford_pets import OxfordPetsSegmentation, collate_fn
from common.train import fit, evaluate
from common.segmenter import UNet, Segmenter


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
    train_dataset = OxfordPetsSegmentation(DATA_DIR, "train", transform=train_transform)
    test_dataset = OxfordPetsSegmentation(DATA_DIR, "test", transform=train_transform)

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
    x, y, masks = train_batch["image"], train_batch["label"], train_batch["mask"]
    print(f"train images: {x.shape}, {x.dtype}, [{x.min():.2f}, {x.max():.2f}]")
    print(f"train labels: {y.shape}, {y.dtype}, [{y.min()}, {y.max()}]")
    print(f"train masks: {masks.shape}, {masks.dtype}, [{masks.min()}, {masks.max()}]")

    #################################################################
    # Modeling
    #################################################################
    model = UNet(in_channels=3, out_channels=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    seg = Segmenter(model, optimizer, num_classes=3, ignore_index=-1)

    print(f"\n>> Training:")
    history = fit(seg, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)
    # history = fit(seg, train_loader, num_epochs=NUM_EPOCHS)

    print(f"\n>> Evaluation:")
    test_results = evaluate(seg, test_loader)
    print(", ".join([f"{k}={v:.3f}" for k, v in test_results.items()]))

    print(f"\n>> Prediction:")
    test_batch = next(iter(test_loader))
    images = test_batch["image"][:NUM_SAMPLES]
    labels = test_batch["label"][:NUM_SAMPLES]

    preds = seg.predict(images)
    for i in range(NUM_SAMPLES):
        print(f"Target: {labels[i].item()} | Prediction: {preds[i].argmax()}")

    del train_loader
    del test_loader
