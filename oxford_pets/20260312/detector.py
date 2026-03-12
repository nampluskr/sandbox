import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.detection import MeanAveragePrecision
from typing import List, Dict, Any
import torchvision.tv_tensors as tv_tensors


class ManualDetector(nn.Module):
    def __init__(self, num_classes=3, in_channels=3):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.extra_layers = nn.Sequential(
            # [B, 256, 28, 28] → [B, 384, 14, 14]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(384),
            nn.MaxPool2d(2),  # [B, 384, 14, 14]

            # [B, 384, 14, 14] → [B, 512, 7, 7]
            nn.Conv2d(384, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),  # [B, 512, 7, 7]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, 512, 1, 1]
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid(),  # normalized [0,1]
        )
        self.cls_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_weight = 0.5

    def forward(self, images, targets=None):
        feat = self.backbone(images)
        feat = self.extra_layers(feat)
        pooled = self.gap(feat)
        shared_feat = self.shared(pooled)
        pred_bboxes = self.bbox_head(shared_feat)
        pred_logits = self.cls_head(shared_feat)

        if self.training:
            return pred_bboxes, pred_logits

        scores, labels = torch.softmax(pred_logits, dim=-1).max(dim=-1)
        boxes = pred_bbox * torch.tensor([images.shape[-2], images.shape[-1]] * 2, device=pred_bboxes.device)
        return [{
            "boxes": box.unsqueeze(0), 
            "labels": label.unsqueeze(0), 
            "scores": score.unsqueeze(0)
        } for box, label, score in zip(boxes, labels, scores)]


class Detector(nn.Module):
    def __init__(self, model, num_classes, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.bbox_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_weight = 0.5

    def forward(self, images, targets):
        outputs = self.model(images, targets)
        if isinstance(outputs, dict):
            return outputs

        H, W = images.shape[-2:]
        gt_bboxes = torch.stack([t["boxes"][0] for t in targets]) / torch.tensor([W, H, W, H]).to(self.device)
        gt_labels = torch.stack([t["labels"][0] for t in targets])

        pred_bboxes, pred_logits = outputs
        loss_bbox = self.bbox_loss(pred_bboxes, gt_bboxes)
        loss_cls = self.cls_loss(pred_logits, gt_labels)

        return {
            "loss_bbox": loss_bbox * self.bbox_weight,
            "loss_cls": loss_cls,
        }

    def train_step(self, batch):
        self.model.train()
        images = batch["image"].to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch["target"]]

        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            **{f"{k}": v.item() for k, v in loss_dict.items()},
            "batch_size": images.size(0),
        }

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        images = batch["image"].to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch["target"]]

        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        return {
            "total": total_loss.item(),
            **{f"{k}": v.item() for k, v in loss_dict.items()},
            "batch_size": images.size(0),
        }

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        images = batch["image"].to(self.device)
        outputs = self.model(images)
        return outputs


def build_model(num_classes):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == "__main__":
    from oxford_pets import get_dataloader
    from loop import fit, evaluate

    DATA_DIR = "/home/namu/myspace/NAMU/datasets/oxford_pets"
    train_loader = get_dataloader(DATA_DIR, "train", task="detection")
    test_loader = get_dataloader(DATA_DIR, "test", task="detection")

    NUM_EPOCHS = 3

    if 1:
        print(f"\n>> ManualDetector:")
        model = ManualDetector(num_classes=37)
        detector = Detector(model, num_classes=37)
        history = fit(detector, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)
        print("Train History:", history)

    if 1:
        print(f"\n>> FasterRCNN:")
        model = build_model(num_classes=37)
        detector = Detector(model, num_classes=37)
        history = fit(detector, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)
        print("Train History:", history)
