import gc
import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors


def get_class_map(samples):
    labels = sorted(set(sample["label"] for sample in samples))
    return {label: idx for idx, label in enumerate(labels)}


def parse_xml(xml_path):
    boxes = []
    if not os.path.exists(xml_path):
        return boxes
    try:
        tree = ET.parse(xml_path)
        for obj in tree.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text) - 1
            ymin = float(bndbox.find("ymin").text) - 1
            xmax = float(bndbox.find("xmax").text) - 1
            ymax = float(bndbox.find("ymax").text) - 1
            boxes.append([xmin, ymin, xmax, ymax])
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return boxes


def get_samples(data_dir, split="train", exclude_corrupt=True, task="classification"):
    if split == 'train':
        split_file = os.path.join(data_dir, 'annotations', 'trainval.txt')
    elif split == 'test':
        split_file = os.path.join(data_dir, 'annotations', 'test.txt')

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    images_png = [
        "Egyptian_Mau_14", "Egyptian_Mau_139", "Egyptian_Mau_145", "Egyptian_Mau_156",
        "Egyptian_Mau_167", "Egyptian_Mau_177", "Egyptian_Mau_186", "Egyptian_Mau_191",
        "Abyssinian_5", "Abyssinian_34",
    ]
    images_corrupt = ["chihuahua_121", "beagle_116"]
    exclude_list = images_png + images_corrupt if exclude_corrupt else []

    samples = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filename, breed, species  = parts[0], parts[1], parts[2]
            if filename in exclude_list:
                continue
            image_path = os.path.join(data_dir, 'images', f'{filename}.jpg')
            mask_path = os.path.join(data_dir, 'annotations', 'trimaps', f'{filename}.png')
            xml_path = os.path.join(data_dir, 'annotations', 'xmls', f'{filename}.xml')

            if not os.path.exists(image_path):
                continue

            if task in ("segmentation", "seg") and not os.path.exists(mask_path):
                continue
            elif task in ("detection", "det") and not os.path.exists(xml_path):
                continue

            samples.append({
                "image_path": image_path,
                "name": filename.rsplit('_', 1)[0],
                "species": int(species) - 1,    # 0-based (0, 1)
                "breed": int(breed) - 1,        # 0-based (0~36)
                "mask_path": mask_path,
                "xml_path": xml_path,
            })
    return samples


def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        values = [b[key] for b in batch]
        if key in ("image", "mask"):
            result[key] = torch.stack(values)
        elif key in ("target", "name"):
            result[key] = values
        else:
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = torch.tensor(values)
    return result


class OxfordPetsClassification(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.samples = get_samples(data_dir, split, task="classification")
        self.transform = transform
        self.classes = sorted(set(sample["breed"] for sample in self.samples))
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = tv_tensors.Image(image)
        species = torch.tensor(sample["species"]).long()
        breed = torch.tensor(sample["breed"]).long()

        if self.transform:
            image = self.transform(image)
        else:
            image = tv_tensors.ToDtype(torch.float32, scale=True)(image)

        return {
            "image": image,
            "species": species,
            "breed": breed,
            "label": breed,
            "name": sample["name"],
        }


class OxfordPetsSegmentation(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.samples = get_samples(data_dir, split, task="segmentation")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = tv_tensors.Image(image)
        species = torch.tensor(sample["species"]).long()
        breed = torch.tensor(sample["breed"]).long()

        mask = Image.open(sample["mask_path"]).convert("L")
        mask = torch.from_numpy(np.array(mask)).long() - 1  # 0-based
        mask = tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = tv_tensors.ToDtype(torch.float32, scale=True)(image)

        return {
            "image": image,
            "species": species,
            "breed": breed,
            "label": breed,
            "name": sample["name"],
            "mask": mask,
        }


class OxfordPetsDetection(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.samples = get_samples(data_dir, split, task="detection")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = tv_tensors.Image(image)
        species = torch.tensor(sample["species"]).long()
        breed = torch.tensor(sample["breed"]).long()

        boxes_list = parse_xml(sample["xml_path"])
        boxes = tv_tensors.BoundingBoxes(
            boxes_list,
            format="XYXY",
            canvas_size=(image.shape[-2], image.shape[-1])
        )
        labels = torch.tensor([sample["breed"]] * len(boxes_list), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(idx),
        }

        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = tv_tensors.ToDtype(torch.float32, scale=True)(image)

        return {
            "image": image,
            "species": species,
            "breed": breed,
            "name": sample["name"],
            "target": target,
        }


if __name__ == "__main__":
    #####################################################################
    # Datasets
    #####################################################################
    DATA_DIR = "/home/namu/myspace/NAMU/datasets/oxford_pets"

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

    cls_train_dataset = OxfordPetsClassification(DATA_DIR, "train", transform=train_transform)
    seg_train_dataset = OxfordPetsSegmentation(DATA_DIR, "train", transform=train_transform)
    det_train_dataset = OxfordPetsDetection(DATA_DIR, "train", transform=train_transform)

    cls_test_dataset = OxfordPetsClassification(DATA_DIR, "test", transform=test_transform)
    seg_test_dataset = OxfordPetsSegmentation(DATA_DIR, "test", transform=test_transform)
    det_test_dataset = OxfordPetsDetection(DATA_DIR, "test", transform=test_transform)

    #####################################################################
    # Data loaders
    #####################################################################
    train_kwargs = {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": collate_fn,
        "pin_memory": True,
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    test_kwargs = {
        "batch_size": 16,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": True,
        "num_workers": 4,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }

    cls_train_loader = DataLoader(cls_train_dataset, **train_kwargs)
    seg_train_loader = DataLoader(seg_train_dataset, **train_kwargs)
    det_train_loader = DataLoader(det_train_dataset, **train_kwargs)

    cls_test_loader = DataLoader(cls_test_dataset, **test_kwargs)
    seg_test_loader = DataLoader(seg_test_dataset, **test_kwargs)
    det_test_loader = DataLoader(det_test_dataset, **test_kwargs)

    #####################################################################
    # Test data loader: classification
    #####################################################################
    batch = next(iter(cls_train_loader))
    images = batch["image"]
    species = batch["species"]
    breeds = batch["breed"]
    names = batch["name"]

    print(f"\n[Classification]")
    print(f"> dataset: {len(cls_train_loader.dataset)}, batches: {len(cls_train_loader)}")
    print(f"> image: {images.shape}, {images.dtype} [{images.min():.2f}, {images.max():.2f}]")
    print(f"> species: {species.shape}, {species.dtype} [{species.min()}, {species.max()}]")
    print(f"> breed: {breeds.shape}, {breeds.dtype} [{breeds.min()}, {breeds.max()}]")
    print(f"> name: {len(names)}, {type(names)}, {names[:5]}")

    #####################################################################
    # Test data loader: segmentation
    #####################################################################
    batch = next(iter(seg_train_loader))
    images = batch["image"]
    species = batch["species"]
    breeds = batch["breed"]
    names = batch["name"]
    masks = batch["mask"]

    print(f"\n[Segmentation]")
    print(f"> dataset: {len(seg_train_loader.dataset)}, batches: {len(seg_train_loader)}")
    print(f"> image: {images.shape}, {images.dtype} [{images.min():.2f}, {images.max():.2f}]")
    print(f"> species: {species.shape}, {species.dtype} [{species.min()}, {species.max()}]")
    print(f"> breed: {breeds.shape}, {breeds.dtype} [{breeds.min()}, {breeds.max()}]")
    print(f"> name: {len(names)}, {type(names)}, {names[:5]}")
    print(f"> mask : {masks.shape}, {masks.dtype} [{masks.min()}, {masks.max()}] (unique: {masks.unique().tolist()})")

    #####################################################################
    # Test data loader: detection
    #####################################################################
    batch = next(iter(det_train_loader))
    images = batch["image"]
    species = batch["species"]
    breeds = batch["breed"]
    names = batch["name"]
    targets = batch["target"]

    print(f"\n[Detection]")
    print(f"> dataset: {len(det_train_loader.dataset)}, batches: {len(det_train_loader)}")
    print(f"> image: {images.shape}, {images.dtype} [{images.min():.2f}, {images.max():.2f}]")
    print(f"> species: {species.shape}, {species.dtype} [{species.min()}, {species.max()}]")
    print(f"> breed: {breeds.shape}, {breeds.dtype} [{breeds.min()}, {breeds.max()}]")
    print(f"> name: {len(names)}, {type(names)}, {names[:5]}")
    print(f"> target: list of {len(targets)} dicts (boxes: {targets[0]['boxes'].shape}, labels: {targets[0]['labels'].shape})")

    #####################################################################
    # Clear data loaders (persistent_workers=True)
    #####################################################################
    del cls_train_loader
    del seg_train_loader
    del det_train_loader

    del cls_test_loader
    del seg_test_loader
    del det_test_loader

    gc.collect()
