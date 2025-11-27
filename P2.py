import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================================
# 1. OUTPUT FOLDER
# ============================================================
output_dir = "/home/sat3812/Final_project/Output_2"
os.makedirs(output_dir, exist_ok=True)
print("Saving all Person2 outputs to:", output_dir)

# ============================================================
# 2. LOAD NPZ FILES
# ============================================================
dataset_base = "/home/sat3812/Final_project/Dataset/npz"

train = np.load(os.path.join(dataset_base, "train.npz"))
val   = np.load(os.path.join(dataset_base, "val.npz"))
test  = np.load(os.path.join(dataset_base, "test.npz"))

X_train, y_train = train["X"], train["y"]
X_val, y_val     = val["X"],   val["y"]
X_test, y_test   = test["X"],  test["y"]

print("\nLoaded:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# ============================================================
# 3. PREPROCESS
# ============================================================
IMG_SIZE = 96
NUM_CLASSES = 7

# grayscale → RGB
X_train = np.repeat(X_train[..., np.newaxis], 3, axis=-1)
X_val   = np.repeat(X_val[..., np.newaxis], 3, axis=-1)
X_test  = np.repeat(X_test[..., np.newaxis], 3, axis=-1)

# resize to 96×96
X_train = tf.image.resize(X_train, (IMG_SIZE, IMG_SIZE)).numpy()
X_val   = tf.image.resize(X_val, (IMG_SIZE, IMG_SIZE)).numpy()
X_test  = tf.image.resize(X_test, (IMG_SIZE, IMG_SIZE)).numpy()


# ============================================================
# 4. BUILD RESNET18-LIKE MODEL (CUSTOM KERAS IMPLEMENTATION)
# ============================================================

def conv_block(x, filters, stride=1):
    shortcut = x

    x = tf.keras.layers.Conv2D(filters, (3,3), strides=stride, padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, (3,3), strides=1, padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters, (1,1), strides=stride, padding="same")(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def build_resnet18(input_shape=(96,96,3), num_classes=7):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, (7,7), strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=2, padding="same")(x)

    # ResNet18 blocks
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    x = conv_block(x, 128, stride=2)
    x = conv_block(x, 128)

    x = conv_block(x, 256, stride=2)
    x = conv_block(x, 256)

    x = conv_block(x, 512, stride=2)
    x = conv_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

print("\nBuilding ResNet18-like model...")
model = build_resnet18()
model.summary()

# Save summary
with open(os.path.join(output_dir, "resnet18_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# ============================================================
# 5. COMPILE
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 6. TRAIN MODEL
# ============================================================
print("\n🔥 Training ResNet18-like model...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=2
)

# ============================================================
# 7. SAVE TRAINING PLOTS
# ============================================================
def save_plot(history, name):
    plt.figure(figsize=(7,4))
    plt.plot(history.history[name], label="train")
    plt.plot(history.history["val_" + name], label="val")
    plt.legend()
    plt.title("ResNet18 " + name.capitalize())
    plt.savefig(os.path.join(output_dir, f"{name}_resnet18.png"))
    plt.close()

save_plot(history, "accuracy")
save_plot(history, "loss")

print("✔ Saved Accuracy & Loss plots")

# ============================================================
# 8. EVALUATE ON TEST SET
# ============================================================
preds = model.predict(X_test).argmax(axis=1)

report = classification_report(y_test, preds, digits=4)
print("\n📊 Classification Report:")
print(report)

with open(os.path.join(output_dir, "classification_report_resnet18.txt"), "w") as f:
    f.write(report)

# CONFUSION MATRIX
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="magma")
plt.title("ResNet18 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(output_dir, "confusion_matrix_resnet18.png"))
plt.close()

print("\n✔ PERSON 2 COMPLETE — ResNet18 Transfer Learning Finished.")
