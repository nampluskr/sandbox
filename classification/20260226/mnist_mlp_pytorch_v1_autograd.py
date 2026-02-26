import os
import gzip
import numpy as np
import random

import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

#################################################################
# Dataset (numpy)
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

#################################################################
# Functions (torch)
#################################################################
def sigmoid(x):
    return torch.where(x >= 0, 1 / (1 + torch.exp(-x)), torch.exp(x) / (1 + torch.exp(x)))

def sigmoid_grad(x):
    return x * (1 - x)

def softmax(x):
    x_max = torch.max(x, dim=1, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / torch.sum(e_x, dim=1, keepdim=True)

def cross_entropy(preds, targets):
    probs = torch.sum(preds * targets, dim=1)
    return -torch.mean(torch.log(probs + 1e-8))

def accuracy(preds, targets):
    pred_classes = torch.argmax(preds, dim=1)
    true_classes = torch.argmax(targets, dim=1)
    return (pred_classes == true_classes).float().mean()

if __name__ == "__main__":
    print(os.path.basename(__file__))

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = r"E:\datasets\mnist"
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-2
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10

    set_seed(SEED)

    #################################################################
    # Data loading
    #################################################################
    x_train_np = load_images(DATA_DIR, "train")  # (60000, 28, 28)
    y_train_np = load_labels(DATA_DIR, "train")  # (60000,)
    x_test_np = load_images(DATA_DIR, "test")    # (10000, 28, 28)
    y_test_np = load_labels(DATA_DIR, "test")    # (10000,)

    #################################################################
    # Data Preprocessing
    #################################################################
    x_train_np = x_train_np.reshape(-1, 784).astype(np.float32) / 255.0
    y_train_np = one_hot(y_train_np, num_classes=10).astype(np.float32)
    x_test_np = x_test_np.reshape(-1, 784).astype(np.float32) / 255.0
    y_test_np = one_hot(y_test_np, num_classes=10).astype(np.float32)

    x_train = torch.from_numpy(x_train_np)  # (60000, 784)
    y_train = torch.from_numpy(y_train_np)  # (60000, 10)
    x_test = torch.from_numpy(x_test_np)    # (10000, 784)
    y_test = torch.from_numpy(y_test_np)    # (10000, 10)

    #################################################################
    # Modeling (with requires_grad)
    #################################################################
    w1 = torch.randn(784, 256)
    b1 = torch.zeros(256)
    w2 = torch.randn(256, 128)
    b2 = torch.zeros(128)
    w3 = torch.randn(128, 10)
    b3 = torch.zeros(10)

    w1.requires_grad_(True)
    b1.requires_grad_(True)
    w2.requires_grad_(True)
    b2.requires_grad_(True)
    w3.requires_grad_(True)
    b3.requires_grad_(True)

    #################################################################
    # Training (with autograd)
    #################################################################
    print(f"\n>> Training:")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        total_acc = 0.0
        total_size = 0

        indices = torch.randperm(len(x_train))
        for idx in range(0, len(x_train), BATCH_SIZE):
            batch_indices = indices[idx:idx + BATCH_SIZE]
            x = x_train[batch_indices]
            y = y_train[batch_indices]
            batch_size = x.size(0)
            total_size += batch_size

            # Forward propagation
            z1 = torch.matmul(x, w1) + b1
            a1 = sigmoid(z1)
            z2 = torch.matmul(a1, w2) + b2
            a2 = sigmoid(z2)
            z3 = torch.matmul(a2, w3) + b3
            preds = softmax(z3)

            loss = cross_entropy(preds, y)
            acc = accuracy(preds, y)

            # Backward propagation (autograd)
            loss.backward()

            # Update weights (no grad)
            with torch.no_grad():
                w1 -= LEARNING_RATE * w1.grad
                b1 -= LEARNING_RATE * b1.grad
                w2 -= LEARNING_RATE * w2.grad
                b2 -= LEARNING_RATE * b2.grad
                w3 -= LEARNING_RATE * w3.grad
                b3 -= LEARNING_RATE * b3.grad

                w1.grad.zero_()
                b1.grad.zero_()
                w2.grad.zero_()
                b2.grad.zero_()
                w3.grad.zero_()
                b3.grad.zero_()

            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Evaluation
    #################################################################
    print(f"\n>> Evaluation:")

    total_loss = 0.0
    total_acc = 0.0
    total_size = 0

    with torch.no_grad():
        for idx in range(0, len(x_test), BATCH_SIZE):
            x = x_test[idx:idx + BATCH_SIZE]
            y = y_test[idx:idx + BATCH_SIZE]
            batch_size = x.size(0)
            total_size += batch_size

            z1 = torch.matmul(x, w1) + b1
            a1 = sigmoid(z1)
            z2 = torch.matmul(a1, w2) + b2
            a2 = sigmoid(z2)
            z3 = torch.matmul(a2, w3) + b3
            preds = softmax(z3)

            loss = cross_entropy(preds, y)
            acc = accuracy(preds, y)

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

        z1 = torch.matmul(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = torch.matmul(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = torch.matmul(a2, w3) + b3
        preds = softmax(z3)

        for i in range(NUM_SAMPLES):
            print(f"Target: {y[i].argmax().item()} | Prediction: {preds[i].argmax().item()}")
