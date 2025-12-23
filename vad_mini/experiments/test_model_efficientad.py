# experiments/load_stfpm.py
import os, sys
source_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

import os
import numpy as numpy
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as T

from vad_mini.data.datasets import MVTecDataset
from vad_mini.data.dataloaders import get_train_loader, get_test_loader
from vad_mini.data.transforms import get_train_transform, get_test_transform, get_mask_transform

from vad_mini.models.efficientad.torch_model import EfficientAdModel


BACKBONE_DIR = "/home/namu/myspace/NAMU/backbones"
DATASET_DIR = "/home/namu/myspace/NAMU/datasets"
# DATASET_DIR = "/mnt/d/deep_learning/datasets"

DATA_DIR = os.path.join(DATASET_DIR, "mvtec")
CATEGORY = "bottle"
IMG_SIZE = 256
NORMALIZE = False
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomGrayscale, Resize, ToTensor
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    imagenet_dir = os.path.join(DATASET_DIR, "imagenette2")
    data_transforms_imagenet = Compose([
        Resize((IMG_SIZE * 2, IMG_SIZE * 2)),
        RandomGrayscale(p=0.3),
        CenterCrop((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
    ])
    imagenet_dataset = ImageFolder(imagenet_dir, transform=data_transforms_imagenet)
    imagenet_loader = DataLoader(imagenet_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # batch_imagenet = next(iter(imagenet_loader))[0].to(DEVICE)
    # print(batch_imagenet.shape)

    if 1:
        train_dataset = MVTecDataset(
            root_dir=DATA_DIR,
            category=CATEGORY,
            split="train",
            transform=get_train_transform(img_size=IMG_SIZE, normalize=NORMALIZE),
            mask_transform=get_mask_transform(img_size=IMG_SIZE),
        )
        train_loader = get_train_loader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
        )

        model = EfficientAdModel(
            teacher_out_channels=384,
            model_size="small",
            padding=False,
            pad_maps=True,
        ).to(DEVICE)

        teacher_path = os.path.join(BACKBONE_DIR, "efficientad_pretrained_weights", "pretrained_teacher_small.pth")
        model.teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE, weights_only=True))

        model.train()
        batch = next(iter(train_loader))
        images = batch["image"].to(DEVICE)
        batch_imagenet = next(iter(imagenet_loader))[0].to(DEVICE)

        loss_st, loss_ae, loss_stae = model(images, batch_imagenet=batch_imagenet)

        print()
        print(f" > loss_st: {loss_st.item():.4f}")
        print(f" > loss_ae: {loss_ae.item():.4f}")
        print(f" > loss_stae: {loss_stae.item():.4f}")

    if 1:
        test_dataset = MVTecDataset(
            root_dir=DATA_DIR,
            category=CATEGORY,
            split="test",
            transform=get_test_transform(img_size=IMG_SIZE, normalize=NORMALIZE),
            mask_transform=get_mask_transform(img_size=IMG_SIZE),
        )
        test_loader = get_test_loader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
        )

        model = EfficientAdModel(
            teacher_out_channels=384,
            model_size="small",
            padding=False,
            pad_maps=True,
        ).to(DEVICE)

        teacher_path = os.path.join(BACKBONE_DIR, "efficientad_pretrained_weights", "pretrained_teacher_small.pth")
        model.teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE, weights_only=True))

        batch = next(iter(test_loader))
        images = batch["image"].to(DEVICE)

        model.eval()
        predictions = model(images)

        print()
        print(f" > anomaly_map: {predictions['anomaly_map'].shape}")
        print(f" > pred_score: {predictions['pred_score'].shape}")
