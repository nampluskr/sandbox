import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class OxfordPets(Dataset):
    def __init__(self, data_dir, split="train", transform=None, mask_transform=None):
        self.transform = transform
        self.mask_transform = mask_transform

        if split == 'train':
            split_file = os.path.join(data_dir, 'annotations', 'trainval.txt')
        elif split == 'test':
            split_file = os.path.join(data_dir, 'annotations', 'test.txt')

        # https://github.com/tensorflow/models/issues/3134
        images_png = [
            "Egyptian_Mau_14",  "Egyptian_Mau_139", "Egyptian_Mau_145", "Egyptian_Mau_156",
            "Egyptian_Mau_167", "Egyptian_Mau_177", "Egyptian_Mau_186", "Egyptian_Mau_191",
            "Abyssinian_5", "Abyssinian_34",
        ]
        images_corrupt = ["chihuahua_121", "beagle_116"]

        self.samples = []
        with open(split_file) as file:
            for line in file:
                filename, label, *_ = line.strip().split()
                if filename not in images_corrupt + images_png:
                    image_path = os.path.join(data_dir, 'images', filename + ".jpg")
                    mask_path = os.path.join(data_dir, 'annotations', 'trimaps', filename + ".png")
                    xml_path = os.path.join(data_dir, 'annotations', 'xmls', filename + ".xml")
                    self.samples.append({
                        "image_path": image_path,
                        "label": int(label) - 1,
                        "mask_path": mask_path,
                        "xml_path": xml_path,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mask_path = sample["mask_path"]
        mask = Image.open(mask_path).convert("L")
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask -1

        if self.mask_transform:
            mask = mask.unsqueeze(0).float()
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0).long()

        label = sample["label"]
        label = torch.tensor(label).long()
        return {
            "image": image,
            "mask": mask,
            "label": label,
        }
