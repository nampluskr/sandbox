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
import torchvision.transforms as T

from common.utils import set_seed
from data.oxford_pets import OxfordPets
from common.trainer import fit, evaluate
from models.segmenter import UNet, Segmenter


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
        T.Resize((224, 224)), 
        # T.RandomHorizontalFlip(0.5),
        # T.RandomRotation(degrees=10),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
    ])
    train_dataset = OxfordPets(DATA_DIR, "train", transform=train_transform, mask_transform=mask_transform)
    test_dataset = OxfordPets(DATA_DIR, "test", transform=train_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    print(f"\n>> Train Data (Batch):")
    train_batch = next(iter(train_loader))
    x, y = train_batch["image"], train_batch["label"]
    print(f"train images: {x.shape}, {x.dtype}, [{x.min():.2f}, {x.max():.2f}]")
    print(f"train labels: {y.shape}, {y.dtype}, [{y.min()}, {y.max()}]")

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
