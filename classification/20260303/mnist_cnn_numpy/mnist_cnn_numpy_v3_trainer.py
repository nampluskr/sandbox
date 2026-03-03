import os
import gzip
import numpy as np

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

#################################################################
# Functions
#################################################################
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sigmoid_grad(x):
    return x * (1 - x)

def softmax(x):
    # x: (N, num_classes)
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy(preds, targets):
    # preds/targets: (N, num_classes)
    probs = np.sum(preds * targets, axis=1)
    return -np.mean(np.log(probs + 1e-8))

def accuracy(preds, targets):
    # preds/targets: (N, num_classes)
    targets = targets.argmax(axis=1)
    return (preds.argmax(axis=1) == targets).mean()

def softmax_cross_entropy_with_logits(logits, targets):
    # logits: (N, num_classes), targets: (N, num_classes)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    log_probs = logits - np.log(np.sum(exp_logits, axis=1, keepdims=True))
    return -np.mean(np.sum(targets * log_probs, axis=1))

#################################################################
# Modules for MLP
#################################################################
class Module:
    def __init__(self):
        self.params = []
        self.grads = []
        self.training = True

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        scale = np.sqrt(1. / in_features)
        self.w = np.random.randn(in_features, out_features) * scale
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

class Sigmoid(Module):
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

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

    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()

#################################################################
# Modules for CNN
#################################################################
class ReLU(Module):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

def im2col(images, kernel_size, stride, padding):
    if padding > 0:
        images = np.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    B, C, H, W = images.shape
    K = kernel_size
    out_h = (H - K) // stride + 1
    out_w = (W - K) // stride + 1
    cols = np.zeros((B, C, K, K, out_h, out_w))

    for y in range(K):
        y_max = y + stride * out_h
        for x in range(K):
            x_max = x + stride * out_w
            cols[:, :, y, x, :, :] = images[:, :, y:y_max:stride, x:x_max:stride]

    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(B * out_h * out_w, -1), out_h, out_w

def col2im(cols, images_shape, kernel_size, stride, padding):
    B, C, H, W = images_shape
    if padding > 0:
        H_pad, W_pad = H + 2 * padding, W + 2 * padding
        images = np.zeros((B, C, H_pad, W_pad))
    else:
        images = np.zeros((B, C, H, W))
        H_pad, W_pad = H, W

    K = kernel_size
    out_h = (H_pad - K) // stride + 1
    out_w = (W_pad - K) // stride + 1
    cols_reshaped = cols.reshape(B, out_h, out_w, C, K, K).transpose(0, 3, 4, 5, 1, 2)

    for y in range(K):
        for x in range(K):
            images[:, :, y:y + stride * out_h:stride, x:x + stride * out_w:stride] += cols_reshaped[:, :, y, x, :, :]

    if padding > 0:
        images = images[:, :, padding:-padding, padding:-padding]
    return images

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(1. / (in_channels * kernel_size * kernel_size))
        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params = [self.w, self.b]
        self.grads = [self.grad_w, self.grad_b]

        self.x = None
        self.col_cache = None  # (col_x, out_h, out_w)
        self.col_w = None

    def forward(self, x):
        B, C, H, W = x.shape
        self.x = x

        col_x, out_h, out_w = im2col(x, self.kernel_size, self.stride, self.padding)
        self.col_cache = (col_x, out_h, out_w)
        self.col_w = self.w.reshape(self.out_channels, -1)  # (out_c, in_c * K * K)

        out = np.dot(col_x, self.col_w.T) + self.b          # (B*out_h*out_w, out_c)
        out = out.reshape(B, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        B, out_c, out_h, out_w = dout.shape
        col_x, out_h_cache, out_w_cache = self.col_cache
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grad_b[...] = np.sum(dout_flat, axis=0)
        grad_w_flat = np.dot(dout_flat.T, col_x)
        self.grad_w[...] = grad_w_flat.reshape(self.grad_w.shape)
        col_w = self.w.reshape(self.out_channels, -1)  # 재계산
        dcol_x = np.dot(dout_flat, col_w)
        dx = col2im(dcol_x, self.x.shape, self.kernel_size, self.stride, self.padding)
        return dx

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self.mask
        return x

    def backward(self, dout):
        if self.training:
            return dout * self.mask
        return dout

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.cache = None

    def forward(self, x):
        B, C, H, W = x.shape
        col_x, out_h, out_w = im2col(x, self.kernel_size, self.stride, self.padding)
        col_x = col_x.reshape(-1, self.kernel_size * self.kernel_size)

        self.out_h, self.out_w = out_h, out_w
        self.input_shape = x.shape
        self.max_indices = np.argmax(col_x, axis=1)
        output = np.max(col_x, axis=1).reshape(B, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.cache = col_x
        return output

    def backward(self, dout):
        B, C, out_h, out_w = dout.shape
        dout_flat = dout.transpose(0, 2, 3, 1).flatten()

        dcol = np.zeros_like(self.cache)
        dcol[np.arange(self.max_indices.size), self.max_indices] = dout_flat

        dx = col2im(dcol, self.input_shape, self.kernel_size, self.stride, self.padding)
        return dx

class Flatten(Module):
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)

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
# Training
#####################################################################
class CrossEntropyWithLogits:
    def __call__(self, logits, targets):
        self.preds = softmax(logits)
        self.targets = targets
        return cross_entropy(self.preds, self.targets)

    def grad(self):
        batch_size = self.preds.shape[0]
        return (self.preds - self.targets) / batch_size

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, x, y):
        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        dout = self.loss_fn.grad()
        model.backward(dout)
        optimizer.step()

        preds = softmax(logits)
        acc = accuracy(preds, y)
        return loss, acc

    def eval_step(self, x, y):
        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        preds = softmax(logits)
        acc = accuracy(preds, y)
        return loss, acc

    def predict(self, x):
        model.eval()
        logits = self.model(x)
        preds = softmax(logits)
        return preds

    def train(self, dataloader):
        model.train()
        total_loss = 0
        total_acc = 0
        total_size = 0

        for x, y in dataloader:
            batch_size = len(x)
            total_size += batch_size

            loss, acc = self.train_step(x, y)
            total_loss += loss * batch_size
            total_acc += acc * batch_size
        return total_loss / total_size, total_acc / total_size

    def evaluate(self, dataloader):
        model.eval()
        total_loss = 0
        total_acc = 0
        total_size = 0

        for x, y in dataloader:
            batch_size = len(x)
            total_size += batch_size

            loss, acc = self.eval_step(x, y)
            total_loss += loss * batch_size
            total_acc += acc * batch_size
        return total_loss / total_size, total_acc / total_size

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
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10

    np.random.seed(SEED)

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
    x_train = x_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0   # (60000, 784)
    y_train = one_hot(y_train, num_classes=10).astype(np.float32)   # (60000, 10)
    x_test = x_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0     # (10000, 784)
    y_test = one_hot(y_test, num_classes=10).astype(np.float32)     # (10000, 10)

    #################################################################
    # Data loaders
    #################################################################
    train_loader = Dataloader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Dataloader(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    #################################################################
    # Modeling
    #################################################################
    model = Sequential(
        Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (N, 16, 28, 28)
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),                 # (N, 16, 14, 14)
        Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (N, 32, 14, 14)
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),                 # (N, 32, 7, 7)
        Flatten(),
        Dropout(p=0.5),
        Linear(32 * 7 * 7, 10)
    )
    optimizer = SGD(model, lr=LEARNING_RATE)
    loss_fn = CrossEntropyWithLogits()
    
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

    x = x_test[:NUM_SAMPLES]
    y = y_test[:NUM_SAMPLES]

    preds = trainer.predict(x)

    for i in range(NUM_SAMPLES):
        print(f"Target: {y[i].argmax()} | Prediction: {preds[i].argmax()}")
