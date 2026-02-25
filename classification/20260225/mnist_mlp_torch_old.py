import os
import gzip
import numpy as np
import torch

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def one_hot(y, n_classes):
    return np.eye(n_classes)[y]

def accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1)
    y_true = y_true.argmax(dim=1)
    return torch.eq(y_pred, y_true).float().mean()

# 데이터 로드
data_dir = r"E:\datasets\mnist"
x_train = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
y_train = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
x_test = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
y_test = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")

x_train_scaled = x_train.astype(np.float32).reshape(-1, 28*28) / 255.0
x_test_scaled = x_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train_onehot = one_hot(y_train, n_classes=10).astype(np.float32)
y_test_onehot = one_hot(y_test, n_classes=10).astype(np.float32)

# 텐서로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = torch.tensor(x_train_scaled).to(device)
y_train = torch.tensor(y_train_onehot).to(device)

# 가중치 초기화 (requires_grad=True 필수)
torch.manual_seed(42)
input_size, hidden_size, output_size = 784, 256, 10

w1 = torch.randn(input_size, hidden_size, device=device) * 0.01
w1.requires_grad_(True)
b1 = torch.zeros(hidden_size, device=device, requires_grad=True)
w2 = torch.randn(hidden_size, output_size, device=device) * 0.01
w2.requires_grad_(True)
b2 = torch.zeros(output_size, device=device, requires_grad=True)

# 학습 설정
n_epochs = 10
batch_size = 32
learning_rate = 0.01

for epoch in range(1, n_epochs + 1):
    indices = torch.randperm(len(x_train))
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for i in range(0, len(x_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        x = x_train[batch_idx]
        y = y_train[batch_idx]

        # 순전파
        z1 = x @ w1 + b1
        a1 = torch.sigmoid(z1)
        z2 = a1 @ w2 + b2
        out = torch.softmax(z2, dim=1)

        # 손실: cross_entropy는 logits과 정수 레이블 필요
        target = y.argmax(dim=1)  # one-hot → class index
        loss = torch.nn.functional.cross_entropy(z2, target)

        # 정확도
        acc = accuracy(out, y)

        # 역전파
        loss.backward()  # 그래프 연결 확인

        # 가중치 업데이트
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            b1 -= learning_rate * b1.grad
            w2 -= learning_rate * w2.grad
            b2 -= learning_rate * b2.grad

            # 기울기 초기화
            w1.grad.zero_()
            b1.grad.zero_()
            w2.grad.zero_()
            b2.grad.zero_()

        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

    if epoch % (n_epochs // 10) == 0:
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(f"[{epoch}/{n_epochs}] loss: {avg_loss:.3f} acc: {avg_acc:.3f}")
