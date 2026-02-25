import numpy as np


def one_hot(x, num_classes):
    return np.eye(num_classes)[x]


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def softmax(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy(preds, targets):
    if targets.ndim == 1:
        batch_size = preds.shape[0]
        probs = preds[np.arange(batch_size), targets]
    else:   # one-hot labels
        probs = np.sum(preds * targets, axis=1)
    return -np.mean(np.log(probs + 1e-8))


def accuracy_fn(preds, targets):
    # preds = softmax(logits)
    if targets.ndim == 2:   # one-hot labels
        targets = targets.argmax(axis=1)
    return (preds.argmax(axis=1) == targets).mean()
