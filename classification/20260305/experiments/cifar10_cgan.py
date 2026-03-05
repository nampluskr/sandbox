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

from common.utils import set_seed, create_images, sample_latent, sample_labels, update_history, plot_images
from data.cifar10 import CIFAR10Dataset
from common.trainer import fit, evaluate
from models.cgan import CDiscriminator, CGenerator, ConditionalGAN


if __name__ == "__main__":
    print(f">> {os.path.basename(__file__)}")

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = "/home/namu/myspace/NAMU/datasets/cifar10"
    NUM_CLASSES = 10
    SEED = 42
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 5
    NUM_SAMPLES = 10

    set_seed(SEED)

    #################################################################
    # Data loading
    #################################################################
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])   # [-1, 1] -> nn.Tanh()
    train_dataset = CIFAR10Dataset(DATA_DIR, "train", transform=transform)
    test_dataset = CIFAR10Dataset(DATA_DIR, "test", transform=transform)

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
    IMG_SIZE = 32
    LATENT_DIM = 64
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    BASE = 64

    EMBEDDING_DIM = 10
    EMBEDDING_CHANNELS = 1

    generator = CGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
    )
    discriminator = CDiscriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
        embedding_channels=EMBEDDING_CHANNELS,
    )
    gan = ConditionalGAN(generator, discriminator)

    #################################################################
    # Training
    #################################################################
    NUM_EPOCHS = 10
    TOTAL_EPOCHS = 200
    NUM_SAMPLES = 100

    FILENAME = os.path.splitext(os.path.basename(__file__))[0]
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", FILENAME)
    IMAGE_NAME = FILENAME + ""

    print(f"\n>> Training:")
    noises = sample_latent(NUM_SAMPLES, LATENT_DIM)
    labels = sample_labels(NUM_SAMPLES, NUM_CLASSES)
    history = {}
    epoch = 0

    for _ in range(TOTAL_EPOCHS // NUM_EPOCHS):
        epoch_history = fit(gan, train_loader, num_epochs=NUM_EPOCHS, total_epochs=TOTAL_EPOCHS)
        update_history(history, epoch_history)

        images = create_images(gan.generator, noises, labels=labels)
        epoch = gan.global_epoch
        image_path = os.path.join(OUTPUT_DIR, f"{IMAGE_NAME}_epoch{epoch:03d}.png")
        plot_images(*images, ncols=10, save_path=image_path)
