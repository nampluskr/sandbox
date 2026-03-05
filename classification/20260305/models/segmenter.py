import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, JaccardIndex


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)   # [B, 64, H, W]
        p1 = self.pool(e1)
        e2 = self.enc2(p1)  # [B, 128, H/2, W/2]
        p2 = self.pool(e2)
        e3 = self.enc3(p2)  # [B, 256, H/4, W/4]
        p3 = self.pool(e3)
        e4 = self.enc4(p3)  # [B, 512, H/8, W/8]
        p4 = self.pool(e4)

        b = self.bottleneck(p4)

        u4 = self.upconv4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.upconv3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        return self.final(d1)


class Segmenter(nn.Module):
    def __init__(self, model, optimizer=None, device=None, num_classes=3, ignore_index=-1):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(self.device)
        self.iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(self.device)

    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, dict):
            return outputs['out']
        return outputs

    def train_step(self, batch):
        self.model.train()
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)

        logits = self.forward(images)
        loss = self.loss_fn(logits, masks)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds = logits.argmax(dim=1)
        acc = self.accuracy(preds, masks)
        iou = self.iou(preds, masks)

        return {
            "loss": loss.item(),
            "acc": acc.item(),
            "iou": iou.item(),
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)

        logits = self.forward(images)
        loss = self.loss_fn(logits, masks)

        preds = logits.argmax(dim=1)
        acc = self.accuracy(preds, masks)
        iou = self.iou(preds, masks)

        return {
            "loss": loss.item(),
            "acc": acc.item(),
            "iou": iou.item(),
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        images = images.to(self.device)
        logits = self.forward(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        return probs.cpu(), preds.cpu()
