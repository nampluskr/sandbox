import gc
import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors


def get_class_names():
    return [
        'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound',
        'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair',
        'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter',
        'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin',
        'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland',
        'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard',
        'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx',
        'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier'
    ]

def get_samples(data_dir, split="train", exclude_corrupt=True, task="classification"):
    if split == 'train':
        split_file = os.path.join(data_dir, 'annotations', 'trainval.txt')
    elif split == 'test':
        split_file = os.path.join(data_dir, 'annotations', 'test.txt')
    else:
        raise ValueError("split must be 'train' or 'test'")

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

            filename, label = parts[0], parts[1]
            if filename in exclude_list:
                continue

            image_path = os.path.join(data_dir, 'images', f'{filename}.jpg')
            if not os.path.exists(image_path):
                continue

            sample = {"image_path": image_path, "label": int(label) - 1}

            if task == "classification":
                samples.append(sample)
            elif task == "segmentation":
                mask_path = os.path.join(data_dir, 'annotations', 'trimaps', f'{filename}.png')
                if os.path.exists(mask_path):
                    sample["mask_path"] = mask_path
                    samples.append(sample)
            elif task == "detection":
                xml_path = os.path.join(data_dir, 'annotations', 'xmls', f'{filename}.xml')
                if os.path.exists(xml_path):
                    sample["xml_path"] = xml_path
                    samples.append(sample)
            else:
                raise ValueError(f"Unsupported task: {task}")
    return samples


def parse_xml(xml_path):
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text) - 1
            ymin = float(bndbox.find("ymin").text) - 1
            xmax = float(bndbox.find("xmax").text) - 1
            ymax = float(bndbox.find("ymax").text) - 1
            if xmin >= 0 and ymin >= 0 and xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return boxes


class OxfordPetsClassification(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.samples = get_samples(data_dir, split, task="classification")
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
            image = tv_tensors.ToDtype(torch.float32, scale=True)(image)

        return {
            "image": image,
            "label": torch.tensor(sample["label"]).long(),
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

        # 1: Foreground 2:Background 3: Not classified
        mask = Image.open(sample["mask_path"]).convert("L")
        mask = torch.from_numpy(np.array(mask)).long() - 1  # 0, 1, 2
        mask = tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = tv_tensors.ToDtype(torch.float32, scale=True)(image)

        return {
            "image": image,
            "label": torch.tensor(sample["label"]).long(),
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
        w, h = image.size
        image = tv_tensors.Image(image)

        bbox_list = parse_xml(sample["xml_path"])
        boxes = tv_tensors.BoundingBoxes(
            bbox_list, format="XYXY", canvas_size=image.shape[-2:])

        labels = torch.tensor([sample["label"]] * len(boxes)).long()
        image_id = torch.tensor(idx).long()
        # area = (ymax - ymin) * (xmax - xmin)
        # iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            # "area": area,
            # "iscrowd": iscrowd
        }

        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = tv_tensors.ToDtype(torch.float32, scale=True)(image)

        return {
            "image": image,
            "label": torch.tensor(sample["label"]).long(),
            "target": target,
        }

def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        values = [b[key] for b in batch]

        if key in ["image", "mask"]:
            result[key] = torch.stack(values)
        elif key == "target":
            result[key] = values
        else:
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = torch.tensor(values)
    return result


def get_transforms(split="train", img_size=224):
    transforms = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((img_size, img_size)),
    ]
    if split == "train":
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def get_dataloader(data_dir, split="train", task="classification"):
    transforms = get_transforms(split)

    if task == "classification":
        dataset = OxfordPetsClassification(data_dir, split, transform=transforms)
    elif task == "segmentation":
        dataset = OxfordPetsSegmentation(data_dir, split, transform=transforms)
    elif task == "detection":
        dataset = OxfordPetsDetection(data_dir, split, transform=transforms)
    else:
        raise ValueError(f"Unsupported task: {task}")

    kwargs = {
        "batch_size": 4 if split == "train" else 2,
        "shuffle": split == "train",
        "drop_last": split == "train",
        "collate_fn": collate_fn,
        "pin_memory": True,
        "num_workers": 4,
        "persistent_workers": split == "train",
        "prefetch_factor": 2,
    }
    return DataLoader(dataset, **kwargs)


if __name__ == "__main__":

    DATA_DIR = "/home/namu/myspace/NAMU/datasets/oxford_pets"

    # cls_train_dataset = OxfordPetsClassification(DATA_DIR, "train")
    # seg_train_dataset = OxfordPetsSegmentation(DATA_DIR, "train")
    # det_train_dataset = OxfordPetsDetection(DATA_DIR, "train")
    # print(f"Classification: {len(cls_train_dataset)}")
    # print(f"Segmentatin:    {len(seg_train_dataset)}")
    # print(f"Detection:      {len(det_train_dataset)}")


    cls_train_loader = get_dataloader(DATA_DIR, "train", task="classification")
    cls_test_loader = get_dataloader(DATA_DIR, "test", task="classification")

    print(f"\nClassification:")
    print(f"train: {len(cls_train_loader.dataset)}")
    cls_batch = next(iter(cls_train_loader))
    images, labels = cls_batch["image"], cls_batch["label"]
    print(f"Image: {images.shape}")
    print(f"Label: {labels.shape}")

    seg_train_loader = get_dataloader(DATA_DIR, "train", task="segmentation")
    seg_test_loader = get_dataloader(DATA_DIR, "test", task="segmentation")

    print(f"\nSegmentation:")
    print(f"train: {len(seg_train_loader.dataset)}")
    seg_batch = next(iter(seg_train_loader))
    images, labels, masks = seg_batch["image"], seg_batch["label"], seg_batch["mask"]
    print(f"Image: {images.shape}")
    print(f"Label: {labels.shape}")
    print(f"masks: {masks.shape}")

    det_train_loader = get_dataloader(DATA_DIR, "train", task="detection")
    det_test_loader = get_dataloader(DATA_DIR, "test", task="detection")

    print(f"\nDetection:")
    print(f"train: {len(det_train_loader.dataset)}")
    det_batch = next(iter(det_train_loader))
    images, labels, targets = det_batch["image"], det_batch["label"], det_batch["target"]
    print(f"Image: {images.shape}")
    print(f"Label: {labels.shape}")
