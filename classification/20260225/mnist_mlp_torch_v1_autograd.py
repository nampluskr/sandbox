import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchmetrics import Accuracy

import mnist


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MNIST(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        super().__init__()
        self.images = mnist.load_images(data_dir, split)
        self.labels = mnist.load_labels(data_dir, split)
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).long()
        return image, label



if __name__ == "__main__":

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = r"E:\datasets\mnist"
    SEED = 42
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)

    #################################################################
    # Data loading
    #################################################################
    train_dataset = MNIST(DATA_DIR, "train")
    test_dataset = MNIST(DATA_DIR, "test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #################################################################
    # Modeling
    #################################################################

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = Accuracy(task="multiclass", num_classes=10).to(DEVICE)

    #################################################################
    # Training
    #################################################################
    print(f"\n>> Training:")

    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        total_acc = 0
        total_size = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            batch_size = len(x)
            total_size += batch_size

            # Forward propagation
            logits = model(x)
            loss = loss_fn(logits, y)
            acc = accuracy_fn(logits, y)

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Evaluation
    #################################################################
    print(f"\n>> Evaluation:")

    model.eval()
    total_loss = 0
    total_acc = 0
    total_size = 0

    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        batch_size = len(x)
        total_size += batch_size

        # Forward propagation
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y)
            acc = accuracy_fn(logits, y)

        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size

    print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Prediction
    #################################################################
    print(f"\n>> Prediction:")

    model.eval()
    x, y = next(iter(test_loader))

    with torch.no_grad():
        x = x[:NUM_SAMPLES].to(DEVICE)
        logits = model(x)
        y_preds = torch.softmax(logits, dim=1)

    y_test = y[:NUM_SAMPLES].cpu().numpy()
    y_preds = y_preds.cpu().numpy()

    for i in range(NUM_SAMPLES):
        print(f"Target: {y_test[i]} | Prediction: {y_preds[i].argmax()}")
