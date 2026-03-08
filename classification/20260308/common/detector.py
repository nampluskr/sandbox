import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.detection import MeanAveragePrecision
from typing import List, Dict, Any


class ManualDetector(nn.Module):
    def __init__(self, num_classes=37, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.device = None  # set_device에서 설정

        # Backbone CNN
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
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),  # /2
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_input_dim = 256 * 7 * 7

        # Bounding box head: [x_center, y_center, w, h] (정규화)
        self.box_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),
            nn.Sigmoid()  # 0~1 (정규화된 좌표)
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, images, targets=None):
        device = next(self.parameters()).device
        batch_size = len(images)
        img_h, img_w = images[0].shape[-2], images[0].shape[-1]

        # 이미지 스택
        x = torch.stack(images).to(device)  # (N, 3, H, W)
        x = self.backbone(x)                # (N, 256, H//8, W//8)
        x = self.pool(x)                    # (N, 256, 7, 7)
        x = x.flatten(1)                    # (N, 12544)

        # Box: 정규화된 좌표 출력 (0~1)
        pred_boxes_norm = self.box_head(x)  # (N, 4) → [x_center, y_center, w, h]
        # Scale to pixel space
        scale = torch.tensor([img_w, img_h, img_w, img_h], device=device)
        pred_boxes = pred_boxes_norm * scale  # (N, 4)

        # Class logits
        pred_logits = self.cls_head(x)      # (N, num_classes)
        pred_scores = pred_logits.softmax(dim=1)  # (N, num_classes)
        pred_labels = pred_logits.argmax(dim=1)   # (N,)

        if targets is not None:
            # Training: Compute losses
            target_boxes_list = []
            target_labels_list = []

            for t in targets:
                boxes = t["boxes"]  # (num_boxes, 4)
                labels = t["labels"]  # (num_boxes,)
                # 간단화: 첫 번째 객체만 사용 (단일 객체 가정)
                target_boxes_list.append(boxes[0])
                target_labels_list.append(labels[0])

            target_boxes = torch.stack(target_boxes_list).to(device)  # (N, 4)
            target_labels = torch.stack(target_labels_list).to(device)  # (N,)

            # Normalize targets for L1 loss
            target_boxes_norm = target_boxes / scale

            loss_box = F.l1_loss(pred_boxes_norm, target_boxes_norm)
            loss_cls = F.cross_entropy(pred_logits, target_labels)

            return {
                "loss_box": loss_box,
                "loss_cls": loss_cls,
                "total_loss": loss_box + loss_cls
            }

        else:
            # Inference: List of dicts (Detector 호환)
            result = []
            for i in range(batch_size):
                result.append({
                    "boxes": pred_boxes[i].unsqueeze(0),        # (1, 4)
                    "labels": pred_labels[i].unsqueeze(0),      # (1,)
                    "scores": pred_scores[i, pred_labels[i]].unsqueeze(0)  # (1,)
                })
            return result


class Detector(nn.Module):
    def __init__(self, model, optimizer=None, device=None, num_classes=37, iou_threshold=0.5):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-4)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

        self.map_metric = MeanAveragePrecision(
            # task="multiclass",
            # num_classes=num_classes,
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
        
        # targets: boxes를 [0,1] 정규화
        targets = []
        for t in batch["target"]:
            t_device = {k: v.to(self.device) for k, v in t.items()}
            
            # 이미 tv_tensors.BoundingBoxes라면 data로 접근
            if isinstance(t["boxes"], tv_tensors.BoundingBoxes):
                boxes = t["boxes"].data  # (N, 4) 픽셀 좌표
            else:
                boxes = t["boxes"]
            
            # 픽셀 → 정규화
            h, w = 224, 224
            boxes_norm = boxes / torch.tensor([w, h, w, h], device=self.device)
            t_device["boxes"] = boxes_norm.clamp(0, 1)
            
            targets.append(t_device)

        preds = self.model(images)  # 이미 [0,1] 출력
        self.map_metric.update(preds, targets)

        # Loss 계산
        loss_dict = self.model(images, batch["target"])
        loss = sum(loss for loss in loss_dict.values())

        log_losses = {f"val_{k}": v.item() for k, v in loss_dict.items()}
        log_losses["val_total_loss"] = loss.item()

        return {**log_losses, "batch_size": len(images)}

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        if isinstance(images, list):
            imgs = [img.to(self.device) for img in images]
        else:
            imgs = [images.to(self.device)]

        preds = self.model(imgs, targets=None)

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
