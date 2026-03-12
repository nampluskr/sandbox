import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy, BinaryAccuracy


class ManualClassifier(nn.Module):
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
        self.model.train()
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
        self.model.eval()
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

def build_model(num_classes):
    from torchvision.models import efficientnet_b0

    BACKBONE_DIR = "/home/namu/myspace/NAMU/backbones"
    weight_path = os.path.join(BACKBONE_DIR, "efficientnet_b0_rwightman-7f5810bc.pth")
    state_dict = torch.load(weight_path, map_location='cpu')

    model = efficientnet_b0(weights=None)
    model.load_state_dict(state_dict, strict=False)
    model.classifier = nn.Linear(1280, num_classes)
    return model


if __name__ == "__main__":
    from oxford_pets import get_dataloader
    from loop import fit, evaluate

    DATA_DIR = "/home/namu/myspace/NAMU/datasets/oxford_pets"
    train_loader = get_dataloader(DATA_DIR, "train", task="classification")
    test_loader = get_dataloader(DATA_DIR, "test", task="classification")

    NUM_EPOCHS = 3

    if 1:
        print(f"\n>> Manual Classifier:")
        model = ManualClassifier(num_classes=37, in_channels=3)
        clf = Classifier(model, num_classes=37)
        history = fit(clf, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)
        print("Train History:", history)

    if 1:
        print(f"\n>> EfficientNet B0:")
        model = build_model(num_classes=37)
        clf = Classifier(model, num_classes=37)
        history = fit(clf, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)
        print("Train History:", history)

    del train_loader
    del test_loader
