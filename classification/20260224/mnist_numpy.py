import numpy as np

from mnist import load_images, load_labels

class Dataloader:
    def __init__(self, images, labels, batch_size, shuffle=False, drop_last=False):
        self.images = np.array(images)
        self.labels = np.array(labels)
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
            yield {
                "image": self.images[idx],
                "label": self.labels[idx],
            }

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
        self.w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
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

def accuracy(preds, targets):
    preds = preds.argmax(axis=1)
    if targets.ndim == 2:   # one-hot labels
        targets = targets.argmax(axis=1)
    return (preds == targets).mean()

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


np.random.seed(42)

x_train = load_images(r"E:\datasets\mnist", "train")
y_train = load_labels(r"E:\datasets\mnist", "train")

x_test = load_images(r"E:\datasets\mnist", "test")
y_test = load_labels(r"E:\datasets\mnist", "test")

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
y_train = one_hot(y_train, num_classes=10).astype(np.float64)

x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
y_test = one_hot(y_test, num_classes=10).astype(np.float64)

train_loader = Dataloader(x_train, y_train, batch_size=128, shuffle=True)


NUM_EPOCHS = 10
LEARNING_RATE = 1e-2

model = Sequential(
    Linear(784, 256),
    Sigmoid(),
    Linear(256, 128),
    Sigmoid(),
    Linear(128, 10),
)
loss_fn = CrossEntropyWithLogits()

for epoch in range(1, NUM_EPOCHS + 1):
        batch_loss = 0
        batch_acc = 0
        total_size = 0

        for batch in train_loader:
            x, y = batch["image"], batch["label"]
            x_size = x.shape[0]
            total_size += x_size

            # Forward propagation
            logits = model(x)
            loss = loss_fn(logits, y)
            acc = accuracy(softmax(logits), y)

            # Backward propagation
            dout = loss_fn.grad()
            model.backward(dout)
            
            # Update weights and biases
            for param, grad in zip(model.params, model.grads):
                param -= LEARNING_RATE * grad

            batch_loss += loss * x_size
            batch_acc += acc * x_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss:{batch_loss/total_size:.3f} acc:{batch_acc/total_size:.3f}")


class SGD:
    def __init__(self, model, lr):
        self.params = model.params
        self.grads = model.grads
        self.lr = lr

    def step(self):
        for param, grad in zip(self.params, self.grads):
            param -= self.lr * grad


NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
)
loss_fn = CrossEntropyWithLogits()
optimizer = SGD(model, lr=LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS + 1):
        batch_loss = 0
        batch_acc = 0
        total_size = 0

        for batch in train_loader:
            x, y = batch["image"], batch["label"]
            x_size = x.shape[0]
            total_size += x_size

            # Forward propagation
            logits = model(x)
            loss = loss_fn(logits, y)
            acc = accuracy(softmax(logits), y)

            # Backward propagation
            dout = loss_fn.grad()
            model.backward(dout)

            # Update weights and biases
            optimizer.step()

            batch_loss += loss * x_size
            batch_acc += acc * x_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss:{batch_loss/total_size:.3f} acc:{batch_acc/total_size:.3f}")


class Adam:
    def __init__(self, model, lr, beta1=0.9, beta2=0.999):
        self.params = model.params
        self.grads = model.grads
        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.ms = [np.zeros_like(param) for param in self.params]
        self.vs = [np.zeros_like(param) for param in self.params]

    def step(self):
        self.iter += 1
        for param, grad, m, v in zip(self.params, self.grads, self.ms, self.vs):
            m[...] = self.beta1 * m + (1 - self.beta1) * grad
            v[...] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self.iter)
            v_hat = v / (1.0 - self.beta2 ** self.iter)

            param[...] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
)
loss_fn = CrossEntropyWithLogits()
optimizer = Adam(model, lr=LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS + 1):
        batch_loss = 0
        batch_acc = 0
        total_size = 0

        for batch in train_loader:
            x, y = batch["image"], batch["label"]
            x_size = x.shape[0]
            total_size += x_size

            # Forward propagation
            logits = model(x)
            loss = loss_fn(logits, y)
            acc = accuracy(softmax(logits), y)

            # Backward propagation
            dout = loss_fn.grad()
            model.backward(dout)

            # Update weights and biases
            optimizer.step()

            batch_loss += loss * x_size
            batch_acc += acc * x_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss:{batch_loss/total_size:.3f} acc:{batch_acc/total_size:.3f}")


def train_step(model, optimizer, x, y):
    pass
