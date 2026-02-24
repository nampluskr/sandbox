import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

print(f"devices: {jax.devices()}")

from mnist import load_images, load_labels, get_class_names

x_train = load_images(r"E:\datasets\mnist", "train")
y_train = load_labels(r"E:\datasets\mnist", "train")

x_test = load_images(r"E:\datasets\mnist", "test")
y_test = load_labels(r"E:\datasets\mnist", "test")

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
y_train = y_train.astype(np.int64)

x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
y_test = y_test.astype(np.int64)

class MLP(nnx.Module):
    def __init__(self, num_classes, rngs):
        self.fc1 = nnx.Linear(784, 256, rngs=rngs)
        self.drop1 = nnx.Dropout(0.3, rngs=rngs)
        self.fc2 = nnx.Linear(256, 128, rngs=rngs)
        self.drop2 = nnx.Dropout(0.3, rngs=rngs)
        self.fc3 = nnx.Linear(128, num_classes, rngs=rngs)
        
    def __call__(self, x):
        x = self.drop1(nnx.relu(self.fc1(x)))
        x = self.drop2(nnx.relu(self.fc2(x)))
        return self.fc3(x)
    
def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

def accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)

@nnx.jit
def train_step(model, optimizer, images, labels):
    def loss_fn(model):
        logits = model(images)
        return cross_entropy_loss(logits, labels), logits
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss, accuracy(logits, labels)

@nnx.jit
def eval_step(model, images, labels):
    logits = model(images)
    return cross_entropy_loss(logits, labels), accuracy(logits, labels)

def get_batches(x, y, batch_size, shuffle=False):
    n = len(y)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start: start + batch_size]
        yield x[idx], y[idx]

def train_epoch(model, optimizer, x, y, batch_size):
    model.train()
    total_loss, total_acc, num_batches = 0, 0, 0
    for images, labels in get_batches(x, y, batch_size, shuffle=True):
        loss, acc = train_step(model, optimizer, images, labels)
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
    return total_loss / num_batches, total_acc / num_batches

def eval_epoch(model, x, y, batch_size):
    model.eval()
    total_loss, total_acc, num_batches = 0, 0, 0
    for images, labels in get_batches(x, y, batch_size, shuffle=False):
        loss, acc = eval_step(model, images, labels)
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
    return total_loss / num_batches, total_acc / num_batches

NUM_CLASSES = 10
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 128

model = MLP(num_classes=NUM_CLASSES, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LEARNING_RATE), wrt=nnx.Param)

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, optimizer, x_train, y_train, BATCH_SIZE)
    valid_loss, valid_acc = eval_epoch(model, x_test, y_test, BATCH_SIZE)
    
    print(f"[{epoch:>2}/{NUM_EPOCHS}] loss:{train_loss:.4f}, acc:{train_acc:.4f}")
    
test_loss, test_acc = eval_epoch(model, x_test, y_test, BATCH_SIZE)
print(f"[Test] loss:{test_loss:.4f}, acc:{test_acc:.4f}")

model.eval()
logits = model(x_test[:5])
preds = jnp.argmax(logits, axis=-1)
for i in range(5):
    print(f"Target: {y_test[i]} : Prediction: {preds[i]}")
  
