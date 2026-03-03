import os
import gzip
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy


#################################################################
# Dataset: CIFAR10
#################################################################
TRAIN_BATCHES  = [f"data_batch_{i}" for i in range(1, 6)]
TEST_BATCHES   = ["test_batch"]

def _load_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    if b'data' in batch:
        batch = {k.decode('utf-8'): v for k, v in batch.items()}
    return batch

def load_images(data_dir, split="train"):
    filenames = TRAIN_BATCHES if split == "train" else TEST_BATCHES
    images_list = []
    for filename in filenames:
        batch = _load_batch(os.path.join(data_dir, "cifar-10-batches-py", filename))
        imgs = batch['data'].reshape(-1, 3, 32, 32)
        images_list.append(imgs)
    images = np.concatenate(images_list, axis=0)
    return images.transpose(0, 2, 3, 1)

def load_labels(data_dir, split="train"):
    filenames = TRAIN_BATCHES if split == "train" else TEST_BATCHES
    labels_list = []
    for filename in filenames:
        batch = _load_batch(os.path.join(data_dir, "cifar-10-batches-py", filename))
        labels_list.extend(batch['labels'])
    return np.array(labels_list, dtype='int32')

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.images = load_images(data_dir, split)
        self.labels = load_labels(data_dir, split)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()

        label = torch.tensor(self.labels[idx]).long()
        return image, label

#################################################################
# Training
#################################################################
class Trainer:
    def __init__(self, model, optimizer, loss_fn, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accuracy = Accuracy(task="multiclass", num_classes=10).to(self.device)

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
        self.model.eval()
        logits = self.model(x)
        preds = torch.softmax(logits, dim=1)
        return preds

    def train(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_size = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
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
            x, y = x.to(self.device), y.to(self.device)
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
    DATA_DIR = r"E:\datasets\cifar10"
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
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
    train_dataset = CIFAR10Dataset(DATA_DIR, "train")
    test_dataset = CIFAR10Dataset(DATA_DIR, "test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # #################################################################
    # # Modeling
    # #################################################################
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
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

        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10)
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
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
    x, y = x.to(DEVICE), y.to(DEVICE)

    preds = trainer.predict(x)

    for i in range(NUM_SAMPLES):
        print(f"Target: {y[i].cpu()} | Prediction: {preds[i].cpu().argmax()}")
