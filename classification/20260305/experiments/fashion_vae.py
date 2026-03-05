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
from data.mnist import MNISTDataset
from models.vae import EncoderSmall, DecoderSmall, Encoder, Decoder, VAE
from common.trainer import fit, evaluate


if __name__ == "__main__":
    print(f">> {os.path.basename(__file__)}")

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = "/home/namu/myspace/NAMU/datasets/fashion_mnist"
    NUM_CLASSES = 10
    SEED = 42
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10

    set_seed(SEED)

    #################################################################
    # Data loading
    #################################################################
    # transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])   # [-1, 1] -> nn.Tanh()
    train_dataset = MNISTDataset(DATA_DIR, "train", padding=True)
    test_dataset = MNISTDataset(DATA_DIR, "test", padding=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

    print(f"\n>> Train Data (Batch):")
    train_batch = next(iter(train_loader))
    x, y = train_batch["image"], train_batch["label"]
    print(f"train images: {x.shape}, {x.dtype}, [{x.min():.2f}, {x.max():.2f}]")
    print(f"train labels: {y.shape}, {y.dtype}, [{y.min()}, {y.max()}]")

    #################################################################
    # Modeling
    #################################################################
    encoder1 = EncoderSmall(latent_dim=2, in_channels=1)
    decoder1 = DecoderSmall(latent_dim=2, out_channels=1)
    ae1 = VAE(encoder1, decoder1, beta=10)

    print(f"\n>> Training:")
    fit(ae1, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)

    encoder2 = Encoder(latent_dim=2, in_channels=1)
    decoder2 = Decoder(latent_dim=2, out_channels=1)
    ae2 = VAE(encoder2, decoder2, beta=10)

    print(f"\n>> Training:")
    fit(ae2, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)
