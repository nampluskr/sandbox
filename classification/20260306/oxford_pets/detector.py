import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanAveragePrecision
from typing import Dict, List, Any


import torch
import torch.nn as nn

class ManualDetector(nn.Module):
    def __init__(self, num_classes=37, in_channels=3):
        super().__init__()
        self.num_classes = num_classes

        # 공���된 CNN 백본
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),  # /8

            nn.AdaptiveAvgPool2d((4, 4)),  # 고정 출력 크기
            nn.Flatten(),  # 256 * 4 * 4 = 4096
        )

        # 백본 출력 크기
        self.fc_input_dim = 256 * 4 * 4

        # Bounding box 회귀 (4개 좌표)
        self.box_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),
            nn.Sigmoid()  # 0~1 정규화 → 후처리에서 이미지 크기로 스케일
        )

        # 클래스 분류
        self.cls_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        x: (N, 3, H, W)
        returns:
            boxes: (N, 4)  [0~1 normalized]
            logits: (N, num_classes)
        """
        features = self.backbone(x)
        boxes = self.box_head(features)  # (N, 4), 값: 0~1
        logits = self.cls_head(features)  # (N, num_classes)
        return boxes, logits


class Detector(nn.Module):
    def __init__(self, model, optimizer=None, device=None, num_classes=37, iou_threshold=0.5):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-4)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

        self.map_metric = MeanAveragePrecision(
            task="multiclass",
            num_classes=num_classes,
            iou_thresholds=[iou_threshold],
            class_metrics=False,
        ).to(self.device)

    def train_step(self, batch):
        self.model.train()
        images = [img.to(self.device) for img in batch["image"]]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch["target"]]

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        log_losses = {k: v.item() for k, v in loss_dict.items()}
        log_losses["total_loss"] = loss.item()

        return {
            **log_losses,
            "batch_size": len(images)
        }

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        images = [img.to(self.device) for img in batch["image"]]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch["target"]]

        preds = self.model(images)
        self.map_metric.update(preds, targets)

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        log_losses = {f"val_{k}": v.item() for k, v in loss_dict.items()}
        log_losses["val_total_loss"] = loss.item()

        return {
            **log_losses,
            "batch_size": len(images)
        }

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        if isinstance(images, list):
            imgs = [img.to(self.device) for img in images]
        else:
            imgs = [images.to(self.device)]

        preds = self.inference(imgs)

        for p in preds:
            p["boxes"] = p["boxes"].cpu()
            p["labels"] = p["labels"].cpu()
            p["scores"] = p["scores"].cpu()

        return preds

    @torch.no_grad()
    def compute_map(self):
        metrics = self.map_metric.compute()
        self.map_metric.reset()
        return {
            "mAP": metrics["map"].item(),
            "mAR": metrics["mar_100"].item(),
        }
