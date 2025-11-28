# =====================================================================
# PERSON 1 — Spark + NPZ Loading + EDA + Baseline CNN 
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
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

# Create folders
os.makedirs(output_dir, exist_ok=True)
plots_dir = os.path.join(output_dir, "plots")
samples_dir = os.path.join(output_dir, "samples")
reports_dir = os.path.join(output_dir, "reports")
model_dir = os.path.join(output_dir, "model")

for d in [plots_dir, samples_dir, reports_dir, model_dir]:
    os.makedirs(d, exist_ok=True)

print("NPZ path:", npz_path)
print("Output directory:", output_dir)

# =====================================================================
# 2. START SPARK (MULTI-NODE CLUSTER)
# =====================================================================
spark = SparkSession.builder \
    .appName("NPZ-Preprocessing-baseline") \
    .master("spark://hadoop1:7077") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

print("Spark started on cluster master:", spark.sparkContext.master)

# =====================================================================
# 3. LOAD NPZ FILES
# =====================================================================
train_npz = np.load(os.path.join(npz_path, "train.npz"))
val_npz   = np.load(os.path.join(npz_path, "val.npz"))
test_npz  = np.load(os.path.join(npz_path, "test.npz"))

X_train, y_train = train_npz["X"], train_npz["y"]
X_val,   y_val   = val_npz["X"],   val_npz["y"]
X_test,  y_test  = test_npz["X"],  test_npz["y"]

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)

# =====================================================================
# 4. NORMALIZE + RESHAPE FOR CNN
# =====================================================================
IMG_SIZE = 48

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Reshaped for CNN:", X_train.shape)

# =====================================================================
# 5. EMOTION LABEL MAP
# =====================================================================
EMOTION_MAP = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "neutral", 5: "sad", 6: "surprise"
}

# =====================================================================
# 6. CLASS DISTRIBUTION PLOT
# =====================================================================
def plot_class_distribution(y, filename):
    count = Counter(y)
    labels = list(EMOTION_MAP.values())
    values = [count.get(i, 0) for i in range(len(labels))]

    plt.figure(figsize=(8,5))
    plt.bar(labels, values, color="steelblue")
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

plot_class_distribution(y_train, "class_distribution.png")

# =====================================================================
# 7. SAMPLE IMAGES GRID
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
    plt.savefig(os.path.join(samples_dir, filename))
    plt.close()

save_sample_images(X_train, y_train, "sample_images.png")

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
with open(os.path.join(reports_dir, "cnn_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

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
# 10. SAVE TRAINING CURVES
# =====================================================================
def save_history_plot(history, key, filename):
    plt.figure()
    plt.plot(history.history[key], label="train")
    plt.plot(history.history["val_" + key], label="val")
    plt.title(key.upper())
    plt.legend()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

save_history_plot(history, "accuracy", "baseline_accuracy.png")
save_history_plot(history, "loss", "baseline_loss.png")

# =====================================================================
# 11. TEST EVALUATION
# =====================================================================
y_pred = model.predict(X_test).argmax(axis=1)

report = classification_report(y_test, y_pred, target_names=list(EMOTION_MAP.values()))
with open(os.path.join(reports_dir, "baseline_report.txt"), "w") as f:
    f.write(report)

# =====================================================================
# 12. CONFUSION MATRIX
# =====================================================================
cm = confusion_matrix(y_test, y_pred)
np.save(os.path.join(reports_dir, "baseline_confusion_matrix.npy"), cm)

plt.figure(figsize=(7,7))
plt.imshow(cm, cmap="Blues")
plt.title("Baseline Confusion Matrix")
plt.colorbar()
plt.savefig(os.path.join(plots_dir, "baseline_confusion_matrix.png"))
plt.close()

# =====================================================================
# 13. SAVE BASELINE MODEL
# =====================================================================
model.save(os.path.join(model_dir, "baseline_model.h5"))

# =====================================================================
# 14. STOP SPARK
# =====================================================================
spark.stop()
print("Person 1 completed. Outputs saved to:", output_dir)
