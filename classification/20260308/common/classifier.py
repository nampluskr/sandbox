import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy, BinaryAccuracy


class CNNModel(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.backbone =  nn.Sequential(
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
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_input_dim = 256 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(-1, self.fc_input_dim)
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, model, optimizer=None, device=None, num_classes=10):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        logits = self.model(images)
        loss = self.loss_fn(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = self.accuracy(logits, labels)
        return {"loss": loss.item(), "acc": acc.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)
        return {"loss": loss.item(), "acc": acc.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        images = images.to(self.device)
        logits = self.model(images)
        preds = torch.softmax(logits, dim=1)
        return preds.cpu()


class BinaryClassifier(nn.Module):
    def __init__(self, model, optimizer=None, device=None, num_classes=10):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy().to(self.device)

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        targets = labels.float().unsqueeze(dim=1)

        logits = self.model(images)
        loss = self.loss_fn(logits, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = self.accuracy(logits, targets)
        return {"loss": loss.item(), "acc": acc.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        targets = labels.float().unsqueeze(dim=1)

        logits = self.model(images)
        loss = self.loss_fn(logits, targets)
        acc = self.accuracy(logits, targets)
        return {"loss": loss.item(), "acc": acc.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        images = images.to(self.device)
        logits = self.model(images)
        preds = torch.sigmoid(logits)
        return preds.squeeze(dim=1).cpu()
