import os
from glob import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors


def get_samples(split, data_dir, category):
    samples = []
    categories = [category] if isinstance(category, str) else list(category)
    for category in categories:
        category_dir = os.path.join(data_dir, category)

        if split == "train":
            train_dir = os.path.join(category_dir, "train", "good")
            for image_path in sorted(glob(os.path.join(train_dir, "*.png"))):
                samples.append({
                    "image_path": image_path,
                    "label": 0,
                    "defect_type": "normal",
                    "category": category,
                })

        elif split == "test":
            test_dir = os.path.join(category_dir, "test")
            for defect_type in sorted(os.listdir(test_dir)):
                for image_path in sorted(glob(os.path.join(test_dir, defect_type, "*.png"))):

                    if defect_type == "good":
                        samples.append({
                            "image_path": image_path,
                            "label": 0,
                            "defect_type": "normal",
                            "category": category,
                        })
                    else:
                        samples.append({
                            "image_path": image_path,
                            "label": 1,
                            "defect_type": defect_type,
                            "category": category,
                        })
    return samples


class MVTecDataset(Dataset):
    def __init__(self, split, data_dir, category, transform=None):
        self.samples = get_samples(split, data_dir, category)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = tv_tensors.Image(image)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToDtype(torch.float32, scale=True)(image)

        return {
            "image": image,
            "label": torch.tensor(sample["label"]).long(),
            "defect_type": sample["defect_type"],
            "category": sample["category"],
        }


def get_transform(split, img_size=256, crop_size=None, normalize=True):
    transform = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((img_size, img_size)),
    ]
    if crop_size is not None:
        transform.append(T.CenterCrop((crop_size, crop_size)))
    if split == "train":
        transform.extend([
            T.RandomHorizontalFlip(p=0.5),
        ])
    if normalize:
        transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transform)


def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        values = [b[key] for b in batch]

        if key in ["image", "mask"]:
            result[key] = torch.stack(values)
        elif key in ["defect_type", "category"]:
            result[key] = values
        else:
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = torch.tensor(values)
    return result


def get_dataloader(split, data_dir, category, batch_size=16, img_size=256, crop_size=None, normalize=True):
    transform = get_transform(split, img_size=img_size, crop_size=crop_size, normalize=normalize)
    dataset = MVTecDataset(split, data_dir, category, transform=transform)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": split == "train",
        "drop_last": split == "train",
        "collate_fn": collate_fn,
        "pin_memory": False,
        "num_workers": 0,
        # "persistent_workers": split == "train",
        # "prefetch_factor": 2,
    }
    return DataLoader(dataset, **kwargs)


if __name__ == "__main__":

    DATASET = "mvtec"
    # DATA_DIR = "/home/namu/myspace/NAMU/datasets/mvtec"
    DATA_DIR = "e:\\datasets\\mvtec"
    CATEGORY = ["carpet", "grid", "leather", "tile", "wood"]
    BATCH_SIZE = 16
    IMG_SIZE = 256
    CROP_SIZE = None
    NORMALIZE = True

    train_loader = get_dataloader(
        split="train", 
        data_dir=DATA_DIR,
        category=CATEGORY,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        crop_size=CROP_SIZE,
        normalize=NORMALIZE,
    )
    test_loader = get_dataloader(
        split="test", 
        data_dir=DATA_DIR,
        category=CATEGORY,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        crop_size=CROP_SIZE,
        normalize=NORMALIZE,
    )

    print(f"\n>> Datasets / Dataloaders:")
    print(f"Train: dataset {len(train_loader.dataset)}, batch {len(train_loader)}")
    print(f"Test:  dataset {len(test_loader.dataset)}, batch {len(test_loader)}")

    batch = next(iter(train_loader))
    images, labels, defect_types = batch["image"], batch["label"], batch["defect_type"]

    print(f"\n>> Train batch:")
    print(f"Image: {images.shape}, [{images.min():.2f}, {images.max():.2f}]")
    print(f"Label: {labels.shape}")
    print(f"Defect types: {defect_types}")
    print(f"Categories: {batch['category']}")

    batch = next(iter(test_loader))
    images, labels, defect_types = batch["image"], batch["label"], batch["defect_type"]

    print(f"\n>> Test batch:")
    print(f"Image: {images.shape}, [{images.min():.2f}, {images.max():.2f}]")
    print(f"Label: {labels.shape}")
    print(f"Defect types: {defect_types}")
    print(f"Categories: {batch['category']}")

    del train_loader
    del test_loader
