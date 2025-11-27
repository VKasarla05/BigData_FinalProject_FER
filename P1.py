# =====================================================================
# PERSON 1 — Spark + NPZ Loading + EDA + Baseline CNN (VM Version)
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from pyspark.sql import SparkSession

# =====================================================================
# 1. SET PATHS
# =====================================================================
npz_path = "/home/sat3812/Final_project/Dataset/npz"
output_dir = "/home/sat3812/Final_project/Output_p1"

os.makedirs(output_dir, exist_ok=True)

print("NPZ path:", npz_path)
print("Output directory:", output_dir)

# =====================================================================
# 2. START SPARK
# =====================================================================
spark = SparkSession.builder \
    .appName("Person1-NPZ-Processing") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("\n✔ Spark started on:", spark.sparkContext.master)

# =====================================================================
# 3. LOAD NPZ FILES
# =====================================================================
train_npz = np.load(os.path.join(npz_path, "train.npz"))
val_npz   = np.load(os.path.join(npz_path, "val.npz"))
test_npz  = np.load(os.path.join(npz_path, "test.npz"))

X_train, y_train = train_npz["X"], train_npz["y"]
X_val,   y_val   = val_npz["X"],   val_npz["y"]
X_test,  y_test  = test_npz["X"],  test_npz["y"]

print("\n===== DATA SHAPES =====")
print("Train:", X_train.shape, "| Labels:", y_train.shape)
print("Val:  ", X_val.shape,   "| Labels:", y_val.shape)
print("Test: ", X_test.shape,  "| Labels:", y_test.shape)

# =====================================================================
# 4. NORMALIZE + RESHAPE FOR CNN
# =====================================================================
IMG_SIZE = 48

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("\n✔ Reshaped for CNN:", X_train.shape)

# =====================================================================
# 5. CLASS LABELS
# =====================================================================
EMOTION_MAP = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "neutral", 5: "sad", 6: "surprise"
}

# =====================================================================
# 6. EDA — CLASS DISTRIBUTION
# =====================================================================
def plot_class_distribution(y, filename):
    count = Counter(y)
    labels = list(EMOTION_MAP.values())
    values = [count.get(i, 0) for i in range(len(labels))]

    plt.figure(figsize=(8,5))
    plt.bar(labels, values, color="skyblue")
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_class_distribution(y_train, "class_distribution.png")
print("✔ Saved:", "class_distribution.png")

# =====================================================================
# 7. EDA — SAMPLE IMAGES GRID
# =====================================================================
def save_sample_images(X, y, filename):
    plt.figure(figsize=(6,6))
    idx = np.random.choice(len(X), 9, replace=False)

    for i, index in enumerate(idx):
        plt.subplot(3,3,i+1)
        plt.imshow(X[index].reshape(48,48), cmap="gray")
        plt.title(EMOTION_MAP[y[index]])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

save_sample_images(X_train, y_train, "sample_images.png")
print("✔ Saved:", "sample_images.png")

# =====================================================================
# 8. BASELINE CNN MODEL
# =====================================================================
def build_baseline_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(48,48,1)),
        MaxPooling2D(),

        Conv2D(64, (3,3), activation="relu", padding="same"),
        MaxPooling2D(),

        Conv2D(128, (3,3), activation="relu", padding="same"),
        MaxPooling2D(),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(7, activation="softmax")
    ])
    return model

model = build_baseline_cnn()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save model summary
with open(os.path.join(output_dir, "cnn_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("✔ Saved: cnn_model_summary.txt")

# =====================================================================
# 9. TRAIN BASELINE CNN
# =====================================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64,
    verbose=2
)

# =====================================================================
# 10. TRAINING PLOTS
# =====================================================================
def save_history_plot(history, key, filename):
    plt.figure()
    plt.plot(history.history[key], label="train")
    plt.plot(history.history["val_" + key], label="val")
    plt.title(key.upper())
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

save_history_plot(history, "accuracy", "baseline_accuracy.png")
save_history_plot(history, "loss", "baseline_loss.png")

print("✔ Saved training plots")

# =====================================================================
# 11. TEST EVALUATION
# =====================================================================
y_pred = model.predict(X_test).argmax(axis=1)

# Classification report
report = classification_report(y_test, y_pred, target_names=list(EMOTION_MAP.values()))
with open(os.path.join(output_dir, "baseline_report.txt"), "w") as f:
    f.write(report)

print("\n✔ Classification Report Saved")

# =====================================================================
# 12. CONFUSION MATRIX
# =====================================================================
cm = confusion_matrix(y_test, y_pred)
np.save(os.path.join(output_dir, "baseline_confusion_matrix.npy"), cm)

# Plot confusion matrix
plt.figure(figsize=(7,7))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig(os.path.join(output_dir, "baseline_confusion_matrix.png"))
plt.close()

print("✔ Saved confusion matrix")

# =====================================================================
# 13. STOP SPARK
# =====================================================================
spark.stop()
print("\n🌟 PERSON 1 COMPLETE — All outputs saved to:", output_dir)
