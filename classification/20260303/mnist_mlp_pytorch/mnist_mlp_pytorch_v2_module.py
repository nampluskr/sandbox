import os
import gzip
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    x_train = load_images(DATA_DIR, "train")    # (60000, 28, 28)
    y_train = load_labels(DATA_DIR, "train")    # (60000,)
    x_test = load_images(DATA_DIR, "test")      # (10000, 28, 28)
    y_test = load_labels(DATA_DIR, "test")      # (10000,)

    #################################################################
    # Data Preprocessing
    #################################################################
    x_train_np = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    y_train_np = one_hot(y_train, num_classes=10).astype(np.float32)
    x_test_np = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    y_test_np = one_hot(y_test, num_classes=10).astype(np.float32)

    x_train = torch.from_numpy(x_train_np)          # (60000, 784)
    y_train = torch.from_numpy(y_train_np)          # (60000, 10)
    x_test = torch.from_numpy(x_test_np)            # (10000, 784)
    y_test = torch.from_numpy(y_test_np)            # (10000, 10)

    #################################################################
    # Modeling
    #################################################################
    w1 = torch.randn(784, 256)
    b1 = torch.zeros(256)
    w2 = torch.randn(256, 128)
    b2 = torch.zeros(128)
    w3 = torch.randn(128, 10)
    b3 = torch.zeros(10)

    linear1 = nn.Linear(784, 256)
    linear2 = nn.Linear(256, 128)
    linear3 = nn.Linear(128, 10)

    with torch.no_grad():
        linear1.weight.copy_(w1.T)
        linear1.bias.copy_(b1)
        linear2.weight.copy_(w2.T)
        linear2.bias.copy_(b2)
        linear3.weight.copy_(w3.T)
        linear3.bias.copy_(b3)

    loss_fn = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=10)

    #################################################################
    # Training
    #################################################################
    print(f"\n>> Training:")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        total_acc = 0
        total_size = 0

        indices = torch.randperm(len(x_train))

        for idx in range(0, len(x_train), BATCH_SIZE):
            x = x_train[indices[idx:idx + BATCH_SIZE]]  # (N, 784)
            y = y_train[indices[idx:idx + BATCH_SIZE]]  # (N, 10)
            batch_size = x.size(0)
            total_size += batch_size

            # Forward propagation
            z1 = linear1(x)                             # (N, 256)
            a1 = torch.sigmoid(z1)                      # (N, 256)
            z2 = linear2(a1)                            # (N, 128)
            a2 = torch.sigmoid(z2)                      # (N, 128)
            z3 = linear3(a2)                            # (N, 10)
            logits = z3

            loss = loss_fn(logits, y.argmax(dim=1))     # (N, 10), (N,)
            acc = accuracy(logits, y.argmax(dim=1))     # (N, 10), (N,)

            # Backward propagation (autograd)
            loss.backward()

            # Update weights (no grad)
            with torch.no_grad():
                for layer in [linear1, linear2, linear3]:
                    for param in layer.parameters():
                        param -= LEARNING_RATE * param.grad
                        param.grad.zero_()

            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Evaluaiton
    #################################################################
    print(f"\n>> Evaluation:")

    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        total_size = 0

        for idx in range(0, len(x_test), BATCH_SIZE):
            x = x_test[idx:idx + BATCH_SIZE]
            y = y_test[idx:idx + BATCH_SIZE]
            batch_size = x.size(0)
            total_size += batch_size

            # Forward propagation
            z1 = linear1(x)
            a1 = torch.sigmoid(z1)
            z2 = linear2(a1)
            a2 = torch.sigmoid(z2)
            z3 = linear3(a2)
            logits = z3

            loss = loss_fn(logits, y.argmax(dim=1))
            acc = accuracy(logits, y.argmax(dim=1))

            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size

        print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Prediction
    #################################################################
    print(f"\n>> Prediction:")

    with torch.no_grad():
        x = x_test[:NUM_SAMPLES]
        y = y_test[:NUM_SAMPLES]

        z1 = linear1(x)
        a1 = torch.sigmoid(z1)
        z2 = linear2(a1)
        a2 = torch.sigmoid(z2)
        z3 = linear3(a2)
        logits = z3
        preds = torch.softmax(logits, dim=1)

    for i in range(NUM_SAMPLES):
        print(f"Target: {y[i].argmax()} | Prediction: {preds[i].argmax()}")
