import sys

source_dir = "/home/namu/myspace/NAMU/projects/anomaly_detection_main/src"
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

########################################################
########################################################
import os
import numpy as numpy
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as T

from anomaly_detection.datasets import MVTecDataset
from anomaly_detection.config import load_config


########################################################
########################################################
config = load_config(os.path.join(source_dir, "..", "configs", "paths.yaml"))
train_dataset = MVTecDataset(
    root_dir=config["MVTec_DIR"], 
    category="bottle", 
    split="train",
    transform=T.Compose([T.Resize((256, 256)), T.ToTensor(),]),
    mask_transform=T.Compose([T.Resize((256, 256)), T.ToTensor(),]),
)

data = train_dataset[10]
image = data["image"].permute(1, 2, 0).numpy()
label = data["label"].numpy()
defect_type = data["defect_type"]
mask = None if data["mask"] is None else data["mask"].squeeze().numpy()

print("\n*** Train Dataset:")
print(f">> dataset: {len(train_dataset)}")
print(f">> image: {image.shape}")
print(f">> label: {label}")
print(f">> defect_type: {defect_type}")
print(f">> mask:  {mask if mask is None else mask.shape}")


########################################################
########################################################
test_dataset = MVTecDataset(
    root_dir=config["MVTec_DIR"], 
    category="bottle", 
    split="test",
    transform=T.Compose([T.Resize((256, 256)), T.ToTensor(),]),
    mask_transform=T.Compose([T.Resize((256, 256)), T.ToTensor(),]),
)

data = test_dataset[-10]
image = data["image"].permute(1, 2, 0).numpy()
label = data["label"].numpy()
defect_type = data["defect_type"]
mask = None if data["mask"] is None else data["mask"].squeeze().numpy()

print("\n*** Test Dataset:")
print(f">> dataset: {len(test_dataset)}")
print(f">> image: {image.shape}")
print(f">> label: {label}")
print(f">> defect_type: {defect_type}")
print(f">> mask:  {mask if mask is None else mask.shape}")
