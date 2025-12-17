# src/anomaly_detection/datasets.py
import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class BaseDataset(Dataset):
    def __init__(self, root_dir, category, split, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []

        self.category_dir = os.path.join(root_dir, category)
        self.split_dir = os.path.join(root_dir, category, split)

        if split == "train":
            self._load_train_samples()
        elif split == "test":
            self._load_test_samples()
        else:
            raise ValueError(f"split must be train or test: {split}")

    def _load_train_samples(self):
        raise NotImplementedError

    def _load_test_samples(self):
        raise NotImplementedError

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def _load_mask(self, mask_path):
        if mask_path is None:
            return None

        mask = Image.open(mask_path).convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image": self._load_image(sample["image_path"]),
            "label": torch.tensor(sample["label"]).long(),
            "defect_type": sample["defect_type"],
            "mask": self._load_mask(sample["mask_path"]),
        }


# =========================================================
# MVTec
# =========================================================

class MVTecDataset(BaseDataset):
    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    def _load_train_samples(self):
        good_dir = os.path.join(self.split_dir, "good")
        for image_path in sorted(glob(os.path.join(good_dir, "*.png"))):
            self.samples.append({
                "image_path": image_path,
                "label": 0,
                "defect_type": "good",
                "mask_path": None
            })

    def _load_test_samples(self):
        for defect_type in sorted(os.listdir(self.split_dir)):
            defect_dir = os.path.join(self.split_dir, defect_type)
            for image_path in sorted(glob(os.path.join(defect_dir, "*.png"))):

                if defect_type == "good":
                    self.samples.append({
                        "image_path": image_path,
                        "label": 0,
                        "defect_type": "good",
                        "mask_path": None
                    })
                else:
                    image_name = os.path.basename(image_path)
                    mask_name = os.path.splitext(image_name)[0] + "_mask.png"
                    mask_path = os.path.join(self.category_dir, "ground_truth", defect_type, mask_name)   
                    self.samples.append({
                        "image_path": image_path,
                        "label": 1,
                        "defect_type": defect_type,
                        "mask_path": mask_path
                    })


# =========================================================
# ViSA
# =========================================================

class ViSADataset(BaseDataset):
    def __init__(self, root_dir, category, split, img_size):
        self.category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(self.category_dir):
            raise FileNotFoundError(f"ViSA category not found: {category}")

        super().__init__(root_dir, category, split, img_size)
        self._load()

    def _load(self):
        split_dir = os.path.join(self.category_dir, self.split)
        if not os.path.isdir(split_dir):
            return

        for fname in sorted(os.listdir(split_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(split_dir, fname)
                self.image_paths.append(path)
                self.labels.append(0 if self.split == "train" else 1)


# =========================================================
# BTAD
# =========================================================

class BTADDataset(BaseDataset):
    def __init__(self, root_dir, category, split, img_size):
        self.category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(self.category_dir):
            raise FileNotFoundError(f"BTAD category not found: {category}")

        super().__init__(root_dir, category, split, img_size)

        if split == "train":
            self._load_train()
        else:
            self._load_test()

    def _load_train(self):
        train_dir = os.path.join(self.category_dir, "train", "ok")
        if not os.path.isdir(train_dir):
            return

        for fname in sorted(os.listdir(train_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_paths.append(os.path.join(train_dir, fname))
                self.labels.append(0)

    def _load_test(self):
        test_dir = os.path.join(self.category_dir, "test")
        if not os.path.isdir(test_dir):
            return

        for defect in sorted(os.listdir(test_dir)):
            defect_dir = os.path.join(test_dir, defect)
            if not os.path.isdir(defect_dir):
                continue

            label = 0 if defect == "ok" else 1

            for fname in sorted(os.listdir(defect_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(defect_dir, fname))
                    self.labels.append(label)
