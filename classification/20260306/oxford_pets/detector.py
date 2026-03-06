import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanAveragePrecision
from typing import List, Dict, Any


class ManualDetector(nn.Module):
    def __init__(self, num_classes=37, in_channels=3):
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

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        self.fc_input_dim = 256 * 4 * 4

        self.box_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),
            nn.Sigmoid()  # 0~1 출력
        )

        self.cls_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, targets=None):
        """
        images: List[Tensor] (N, 3, H, W)
        targets: Optional[List[Dict]] for training
        returns:
            - training: dict of losses
            - inference: List[Dict] with 'boxes', 'labels', 'scores'
        """
        device = images[0].device
        batch_size = len(images)
        img_h, img_w = images[0].shape[-2], images[0].shape[-1]

        # Stack images
        x = torch.stack(images)  # (N, 3, H, W)
        features = self.backbone(x)

        # Bounding box (0~1) → scale to pixel
        pred_boxes_norm = self.box_head(features)  # (N, 4)
        pred_boxes = pred_boxes_norm * torch.tensor([img_w, img_h, img_w, img_h], device=device)  # (N, 4)

        # Classification
        pred_logits = self.cls_head(features)  # (N, 37)
        pred_scores = pred_logits.softmax(dim=1)  # (N, 37)
        pred_labels = pred_logits.argmax(dim=1)  # (N,)

        if targets is not None:
            # Training mode: compute losses
            target_boxes_list = []
            target_labels_list = []
            for t in targets:
                target_boxes_list.append(t["boxes"][0])  # (4,)
                target_labels_list.append(t["labels"][0])  # scalar
            target_boxes = torch.stack(target_boxes_list).to(device)  # (N, 4)
            target_labels = torch.stack(target_labels_list).to(device)  # (N,)

            # Normalize target boxes for loss
            target_boxes_norm = target_boxes / torch.tensor([img_w, img_h, img_w, img_h], device=device)

            loss_box = nn.functional.l1_loss(pred_boxes_norm, target_boxes_norm)
            loss_cls = nn.functional.cross_entropy(pred_logits, target_labels)

            return {"loss_box": loss_box, "loss_cls": loss_cls, "total_loss": loss_box + loss_cls}

        else:
            # Inference mode: return list of dicts
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

        preds = self.model(images)  # inference
        self.map_metric.update(preds, targets)

        loss_dict = self.model(images, targets)  # compute loss
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
