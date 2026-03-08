import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, JaccardIndex


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=64):
        super().__init__()
        self.enc_block1 = DoubleConv(in_channels, base)
        self.enc_block2 = DoubleConv(base, base * 2)
        self.enc_block3 = DoubleConv(base * 2, base * 4)
        self.enc_block4 = DoubleConv(base * 4, base * 8)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.upconv4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec_block4 = DoubleConv(base * 16, base * 8)

        self.upconv3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec_block3 = DoubleConv(base * 8, base * 4)

        self.upconv2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec_block2 = DoubleConv(base * 4, base * 2)

        self.upconv1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec_block1 = DoubleConv(base * 2, base)

        self.final_conv = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc_block1(x)
        p1 = self.maxpool(e1)

        e2 = self.enc_block2(p1)
        p2 = self.maxpool(e2)

        e3 = self.enc_block3(p2)
        p3 = self.maxpool(e3)

        e4 = self.enc_block4(p3)
        p4 = self.maxpool(e4)

        b = self.bottleneck(p4)

        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec_block4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec_block3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec_block2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec_block1(d1)

        out = self.final_conv(d1)
        return out

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
