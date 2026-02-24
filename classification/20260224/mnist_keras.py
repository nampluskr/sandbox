import tensorflow as tf
from tensorflow import keras
import numpy as np

print(f"tensorflow: {tf.__version__}")
print(f"keras: {keras.__version__}")
print(f"numpy: {np.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"CPU: {tf.config.list_physical_devices('CPU')}")

from mnist import load_images, load_labels

data_dir = r"E:\datasets\mnist"
x_train = load_images(data_dir, "train")
y_train = load_labels(data_dir, "train")
x_test = load_images(data_dir, "test")
y_test = load_labels(data_dir, "test")

print(f"train: {x_train.shape}, {y_train.shape}")
print(f"test:  {x_test.shape}, {y_test.shape}")

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# x_train = x_train[..., np.newaxis]
# x_test = x_test[..., np.newaxis]

input_shape = (28, 28, 1)
num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential([
    keras.Input(shape=input_shape),
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation="softmax")
])
model.summary()

model.compile(
    optimizer="adam",
    # loss="sparse_categorical_crossentropy",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

predictions = model.predict(x_test[:5], verbose=0)
pred_labels = np.argmax(predictions, axis=1)

for i in range(5):
    print(f"True: {y_test[i].argmax()} | Preds: {pred_labels[i]}")
