import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# ============================================================
# 1. SETUP OUTPUT FOLDER
# ============================================================
output_dir = "/home/sat3812/Final_project/Output_2"
os.makedirs(output_dir, exist_ok=True)
print("Saving all Person2 outputs to:", output_dir)

# ============================================================
# 2. LOAD NPZ FILES (already created by Person1)
# ============================================================
train = np.load("/home/sat3812/Final_project/Dataset/npz/train.npz")
val   = np.load("/home/sat3812/Final_project/Dataset/npz/val.npz")
test  = np.load("/home/sat3812/Final_project/Dataset/npz/test.npz")

X_train, y_train = train["X"], train["y"]
X_val, y_val = val["X"], val["y"]
X_test, y_test = test["X"], test["y"]

print("\nLoaded:")
print("Train:", X_train.shape, "| Labels:", y_train.shape)
print("Val:  ", X_val.shape, "| Labels:", y_val.shape)
print("Test: ", X_test.shape, "| Labels:", y_test.shape)

# ============================================================
# 3. PREPROCESS — ResNet18 expects 3-channel RGB inputs
# ============================================================
IMG_SIZE = 96   # small & fast for CPU
NUM_CLASSES = 7

X_train = np.repeat(X_train[..., np.newaxis], 3, axis=-1)
X_val   = np.repeat(X_val[..., np.newaxis], 3, axis=-1)
X_test  = np.repeat(X_test[..., np.newaxis], 3, axis=-1)

X_train = tf.image.resize(X_train, (IMG_SIZE, IMG_SIZE)).numpy()
X_val   = tf.image.resize(X_val, (IMG_SIZE, IMG_SIZE)).numpy()
X_test  = tf.image.resize(X_test, (IMG_SIZE, IMG_SIZE)).numpy()

# ============================================================
# 4. IMPORT RESNET18 FROM TENSORFLOW HUB
# ============================================================
import tensorflow_hub as hub

print("\n📥 Loading ResNet18 from TF Hub (light & CPU-safe)...")
resnet_url = "https://tfhub.dev/tensorflow/resnet_18/classification/1"

base = hub.KerasLayer(
    resnet_url,
    trainable=False,  # freeze backbone
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# ============================================================
# 5. BUILD TRANSFER LEARNING MODEL
# ============================================================
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = inputs
x = base(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 6. MODEL SUMMARY → SAVE TO TEXT FILE
# ============================================================
summary_path = os.path.join(output_dir, "resnet18_model_summary.txt")
with open(summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("✔ Saved: resnet18_model_summary.txt")

# ============================================================
# 7. TRAIN MODEL (CPU Optimized)
# ============================================================
print("\n🔥 Training ResNet18 Transfer Learning Model...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=2
)

# ============================================================
# 8. PLOTS — ACCURACY & LOSS
# ============================================================
def save_plot():
    # Accuracy plot
    plt.figure(figsize=(7,4))
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("ResNet18 Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_resnet18.png"))
    plt.close()

    # Loss plot
    plt.figure(figsize=(7,4))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("ResNet18 Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_resnet18.png"))
    plt.close()

save_plot()
print("✔ Saved accuracy & loss plots")

# ============================================================
# 9. EVALUATE ON TEST SET
# ============================================================
preds = model.predict(X_test).argmax(axis=1)

print("\n📊 Classification Report:\n")
report = classification_report(y_test, preds, digits=4)
print(report)

with open(os.path.join(output_dir, "classification_report_resnet18.txt"), "w") as f:
    f.write(report)

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
plt.title("ResNet18 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(output_dir, "confusion_matrix_resnet18.png"))
plt.close()

print("\n✔ PERSON 2 COMPLETE — ResNet18 Transfer Learning Finished.")
