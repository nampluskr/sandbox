import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from mnist import load_images, load_labels, get_class_names

class_names = get_class_names()

x_train = load_images(r"E:\datasets\mnist", "train")
y_train = load_labels(r"E:\datasets\mnist", "train")

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
y_train = y_train.astype(np.int64)

x_test = load_images(r"E:\datasets\mnist", "test")
y_test = load_labels(r"E:\datasets\mnist", "test")

x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
y_test = y_test.astype(np.int64)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=128,
    learning_rate_init=1e-3,
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=True,
)
model.fit(x_train, y_train)
print(f"num epochs: {model.n_iter_}")

y_pred = model.predict(x_test)
print("*** Classification Repost ***")
print(classification_report(y_test, y_pred, target_names=class_names))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(model.loss_curve_, label="Train loss")
if model.validation_scores_ is not None:
    val_loss = [1 - s for s in model.validation_scores_]
    axes[0].plot(val_loss, label="Valid loss")
axes[0].legend()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")

fig.tight_layout()
plt.show()

proba = model.predict_proba(x_test[:5])
for i in range(5):
    pred = y_pred[i]
    conf = proba[i][pred] * 100
    print(f"Target: {y_test[i]} | Prediction: {pred} | {conf:.1f}%")

