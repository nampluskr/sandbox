import numpy as np

import mnist
from functions import one_hot, sigmoid, relu, softmax, cross_entropy, accuracy_fn


#####################################################################
# Dataloader
#####################################################################

class Dataloader:
    def __init__(self, images, labels, batch_size, shuffle=False, drop_last=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_images = len(self.images)

        if drop_last:
            self.num_batches = self.num_images // batch_size
        else:
            self.num_batches = (self.num_images + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        indices = np.arange(self.num_images)
        if self.shuffle:
            np.random.shuffle(indices)
        if self.drop_last:
            indices = indices[:self.num_batches * self.batch_size]

        for i in range(self.num_batches):
            idx = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.images[idx], self.labels[idx]


#####################################################################
# Network Layer Modules
#####################################################################

class Module:
    def __init__(self):
        self.params = []
        self.grads = []

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = np.random.randn(in_features, out_features)
        self.b = np.zeros(out_features)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params.extend([self.w, self.b])
        self.grads.extend([self.grad_w, self.grad_b])
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        self.grad_w[...] = np.dot(self.x.T, dout)
        self.grad_b[...] = np.sum(dout, axis=0)
        return np.dot(dout, self.w.T)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)


class ReLU(Module):
    def forward(self, x):
        self.mask = x <= 0
        self.out = relu(x)
        return self.out

    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout


class Sigmoid(Module):
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


#####################################################################
# Optimizers
#####################################################################

class SGD:
    def __init__(self, model, lr):
        self.params = model.params
        self.grads = model.grads
        self.lr = lr

    def step(self):
        for param, grad in zip(self.params, self.grads):
            param -= self.lr * grad


#####################################################################
# Training
#####################################################################

class CrossEntropyWithLogits:
    def __call__(self, logits, targets):
        self.preds = softmax(logits)
        self.targets = targets
        if targets.ndim == 1:
            self.targets = one_hot(targets, logits.shape[1])
        return cross_entropy(self.preds, self.targets)

    def grad(self):
        batch_size = self.preds.shape[0]
        return (self.preds - self.targets) / batch_size


def train_step(model, optimizer, loss_fn, images, labels):
    logits = model(images)
    loss = loss_fn(logits, labels)

    dout = loss_fn.grad()
    model.backward(dout)
    optimizer.step()

    preds = softmax(logits)
    acc = accuracy_fn(preds, labels)
    return loss, acc


def eval_step(model, loss_fn, images, labels):
    logits = model(images)
    loss = loss_fn(logits, labels)

    preds = softmax(logits)
    acc = accuracy_fn(preds, labels)
    return loss, acc


def predict(model, images):
    logits = model(images)
    preds = softmax(logits)
    return preds


if __name__ == "__main__":

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = r"E:\datasets\mnist"
    SEED = 42
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-2
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10

    np.random.seed(SEED)

    #################################################################
    # Data loading
    #################################################################
    x_train = mnist.load_images(DATA_DIR, "train")
    y_train = mnist.load_labels(DATA_DIR, "train")
    x_test = mnist.load_images(DATA_DIR, "test")
    y_test = mnist.load_labels(DATA_DIR, "test")

    #################################################################
    # Data Preprocessing
    #################################################################
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    y_train = one_hot(y_train, num_classes=10).astype(np.float64)
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    y_test = one_hot(y_test, num_classes=10).astype(np.float64)

    train_loader = Dataloader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Dataloader(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    #################################################################
    # Modeling
    #################################################################
    model = Sequential(
        Linear(784, 256),
        Sigmoid(),
        Linear(256, 128),
        Sigmoid(),
        Linear(128, 10),
    )
    optimizer = SGD(model, lr=LEARNING_RATE)
    loss_fn = CrossEntropyWithLogits()

    #################################################################
    # Training
    #################################################################
    print(f"\n>> Training:")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        total_acc = 0
        total_size = 0

        for x, y in train_loader:
            batch_size = len(x)
            total_size += batch_size

            loss, acc = train_step(model, optimizer, loss_fn, x, y)

            total_loss += loss * batch_size
            total_acc += acc * batch_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Evaluation
    #################################################################
    print(f"\n>> Evaluation:")

    total_loss = 0
    total_acc = 0
    total_size = 0

    for x, y in test_loader:
        batch_size = len(x)
        total_size += batch_size

        loss, acc = eval_step(model, loss_fn, x, y)

        total_loss += loss * batch_size
        total_acc += acc * batch_size

    print(f"[{epoch:>2}/{NUM_EPOCHS}] "
          f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Prediction
    #################################################################
    print(f"\n>> Prediction:")

    x = x_test[:NUM_SAMPLES]
    y_preds = predict(model, x)

    for i in range(NUM_SAMPLES):
        print(f"Target: {y_test[i].argmax()} | Prediction: {y_preds[i].argmax()}")

    results = """

>> Training:
[ 1/10] loss:4.630 acc:0.175
[ 2/10] loss:2.577 acc:0.339
[ 3/10] loss:1.993 acc:0.443
[ 4/10] loss:1.681 acc:0.509
[ 5/10] loss:1.484 acc:0.557
[ 6/10] loss:1.344 acc:0.592
[ 7/10] loss:1.238 acc:0.621
[ 8/10] loss:1.154 acc:0.645
[ 9/10] loss:1.086 acc:0.663
[10/10] loss:1.029 acc:0.681

>> Evaluation:
[10/10] loss:1.000 acc:0.682

>> Prediction:
Target: 7 | Prediction: 7
Target: 2 | Prediction: 6
Target: 1 | Prediction: 1
Target: 0 | Prediction: 0
Target: 4 | Prediction: 4
Target: 1 | Prediction: 1
Target: 4 | Prediction: 4
Target: 9 | Prediction: 5
Target: 5 | Prediction: 6
Target: 9 | Prediction: 9
"""
