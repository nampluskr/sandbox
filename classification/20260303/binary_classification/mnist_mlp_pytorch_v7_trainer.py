import os
import gzip
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryAccuracy


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

def to_binary_label(x, positive_class=0):
    binary = (x == positive_class)
    return binary.reshape(-1, 1)

class MNISTDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.images = load_images(data_dir, split)
        self.labels = load_labels(data_dir, split)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.reshape(784).astype(np.float32) / 255.0

        label = self.labels[idx]
        label = to_binary_label(label, positive_class=0).reshape(1).astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(label)  # (784,), (10,)

#################################################################
# Training
#################################################################
class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accuracy = BinaryAccuracy()

    def train_step(self, x, y):
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, acc

    @torch.no_grad()
    def eval_step(self, x, y):
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        return loss, acc

    @torch.no_grad()
    def predict(self, x):
        model.eval()
        logits = self.model(x)
        preds = torch.sigmoid(logits)
        return preds

    def train(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_size = 0

        for x, y in dataloader:
            batch_size = x.size(0)
            total_size += batch_size

            loss, acc = self.train_step(x, y)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
        return total_loss/total_size, total_acc/total_size

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_size = 0

        for x, y in dataloader:
            batch_size = x.size(0)
            total_size += batch_size

            loss, acc = self.eval_step(x, y)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
        return total_loss/total_size, total_acc/total_size

    def fit(self, train_loader, num_epochs, valid_loader=None):
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train(train_loader)

            if valid_loader is not None:
                valid_loss, valid_acc = self.evaluate(valid_loader)
                print(f"[{epoch:>2}/{num_epochs}] "
                    f"loss={train_loss:.3f}, acc={train_acc:.3f}"
                    f" | (val) loss={valid_loss:.3f}, acc={valid_acc:.3f}")
            else:
                print(f"[{epoch:>2}/{num_epochs}] "
                    f"loss={train_loss:.3f}, acc={train_acc:.3f}")

if __name__ == "__main__":
    print(f">> {os.path.basename(__file__)}")

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = r"E:\datasets\mnist"
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-2
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10

    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
    w1 = torch.randn(784, 256)
    b1 = torch.zeros(256)
    w2 = torch.randn(256, 128)
    b2 = torch.zeros(128)
    w3 = torch.randn(128, 1)
    b3 = torch.zeros(1)

    linear1 = nn.Linear(784, 256)
    linear2 = nn.Linear(256, 128)
    linear3 = nn.Linear(128, 1)

    with torch.no_grad():
        linear1.weight.copy_(w1.T)
        linear1.bias.copy_(b1)
        linear2.weight.copy_(w2.T)
        linear2.bias.copy_(b2)
        linear3.weight.copy_(w3.T)
        linear3.bias.copy_(b3)

    model = nn.Sequential(
        linear1, nn.Sigmoid(),
        linear2, nn.Sigmoid(),
        linear3,
    )
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    trainer = Trainer(model, optimizer, loss_fn)

    #################################################################
    # Training
    #################################################################
    print(f"\n>> Training:")

    # for epoch in range(1, NUM_EPOCHS + 1):
    #     train_loss, train_acc = trainer.train(train_loader)
    #     print(f"[{epoch:>2}/{NUM_EPOCHS}] loss:{train_loss:.3f} acc:{train_acc:.3f}")

    trainer.fit(train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)

    #################################################################
    # Evaluaiton
    #################################################################
    print(f"\n>> Evaluation:")

    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"loss:{test_loss:.3f} acc:{test_acc:.3f}")

    #################################################################
    # Prediction
    #################################################################
    print(f"\n>> Prediction:")

    x_test, y_test = next(iter(test_loader))
    x = x_test[:NUM_SAMPLES]
    y = y_test[:NUM_SAMPLES]

    preds = trainer.predict(x)
    pred_labels = (preds >= 0.5).int()

    for i in range(NUM_SAMPLES):
        print(f"Target: {y[i].item()} | Prediction: {pred_labels[i].item()} (prob: {preds[i].item():.3f})")
