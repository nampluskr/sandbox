import os
import gzip
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy


#################################################################
# Dataset
#################################################################
def load_images(data_dir, split="train"):
    filename = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28).copy()

def load_labels(data_dir, split="train"):
    filename = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.copy()

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]

class MNISTDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.images = load_images(data_dir, split)
        self.labels = load_labels(data_dir, split)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.reshape(1, 28, 28).astype(np.float32) / 255.0

        label = self.labels[idx]
        label = one_hot(label, num_classes=10).astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(label)  # (784,), (10,)

if __name__ == "__main__":
    print(f">> {os.path.basename(__file__)}")

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = r"E:\datasets\mnist"
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    #################################################################
    # Data loading
    #################################################################
    train_dataset = MNISTDataset(DATA_DIR, "train")
    test_dataset = MNISTDataset(DATA_DIR, "test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #################################################################
    # Modeling
    #################################################################
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(32 * 7 * 7, 10),
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=10).to(DEVICE)

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
            batch_size = x.size(0)
            total_size += batch_size

            # Forward propagation
            logits = model(x)                           # (N, 10)
            loss = loss_fn(logits, y.argmax(dim=1))     # (N, 10), (N,)
            acc = accuracy(logits, y.argmax(dim=1))     # (N, 10), (N,)

            # Backward propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Evaluaiton
    #################################################################
    print(f"\n>> Evaluation:")

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        total_size = 0

        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            batch_size = x.size(0)
            total_size += batch_size

            # Forward propagation
            logits = model(x)

            loss = loss_fn(logits, y.argmax(dim=1))
            acc = accuracy(logits, y.argmax(dim=1))

            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size

        print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Prediction
    #################################################################
    print(f"\n>> Prediction:")

    x_test, y_test = next(iter(test_loader))

    model.eval()
    with torch.no_grad():
        x = x_test[:NUM_SAMPLES]
        y = y_test[:NUM_SAMPLES]
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        preds = torch.softmax(logits, dim=1)

    for i in range(NUM_SAMPLES):
        print(f"Target: {y[i].cpu().argmax()} | Prediction: {preds[i].cpu().argmax()}")
