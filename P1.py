# ===============================================================
# PERSON 1 (VM VERSION)
# Spark + Load NPZ + EDA + Baseline CNN
# ===============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession

# ---------------------------------------------------------------
# 1. START SPARK SESSION
# ---------------------------------------------------------------
spark = SparkSession.builder \
    .appName("FER_Person1_NPZ_Project") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("✔ Spark started on:", spark.sparkContext.master)

# ---------------------------------------------------------------
# 2. LOAD NPZ FILES FROM VM
# ---------------------------------------------------------------
npz_path = "/home/sat3812/Final_project/Dataset/npz"

train_npz = np.load(f"{npz_path}/train.npz")
val_npz   = np.load(f"{npz_path}/val.npz")
test_npz  = np.load(f"{npz_path}/test.npz")

X_train, y_train = train_npz["X"], train_npz["y"]
X_val,   y_val   = val_npz["X"],   val_npz["y"]
X_test,  y_test  = test_npz["X"],  test_npz["y"]

print("✔ Loaded NPZ files")

# ---------------------------------------------------------------
# 3. PRINT SHAPES
# ---------------------------------------------------------------
print("\n===== DATA SHAPES =====")
print("Train:", X_train.shape, "| Labels:", y_train.shape)
print("Val:  ", X_val.shape,   "| Labels:", y_val.shape)
print("Test: ", X_test.shape,  "| Labels:", y_test.shape)

# ---------------------------------------------------------------
# 4. EDA — Class Distribution
# ---------------------------------------------------------------
emotion_map = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "neutral", 5: "sad", 6: "surprise"
}

def plot_distribution(y, title):
    counts = Counter(y)
    labels = [emotion_map[i] for i in sorted(counts.keys())]
    values = [counts[i] for i in sorted(counts.keys())]

    plt.figure(figsize=(8,5))
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ','_')}.png")
    plt.show()

plot_distribution(y_train, "Train Class Distribution")
plot_distribution(y_val,   "Validation Class Distribution")
plot_distribution(y_test,  "Test Class Distribution")

# ---------------------------------------------------------------
# 5. SAMPLE IMAGES
# ---------------------------------------------------------------
def show_samples(X, y):
    plt.figure(figsize=(6,6))
    indices = np.random.choice(len(X), 9, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(3,3,i+1)
        plt.imshow(X[idx], cmap="gray")
        plt.title(emotion_map[y[idx]])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("Sample_Images.png")
    plt.show()

show_samples(X_train, y_train)

# ---------------------------------------------------------------
# 6. PREPARE FOR CNN
# ---------------------------------------------------------------
X_train_cnn = X_train.reshape(-1, 48, 48, 1)
X_val_cnn   = X_val.reshape(-1, 48, 48, 1)
X_test_cnn  = X_test.reshape(-1, 48, 48, 1)

# ---------------------------------------------------------------
# 7. BASELINE CNN MODEL
# ---------------------------------------------------------------
def get_baseline_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same",
                               input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(7, activation="softmax")
    ])
    return model

model = get_baseline_cnn()
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------------------
# 8. TRAIN BASELINE CNN
# ---------------------------------------------------------------
history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=15,
    batch_size=64
)

# ---------------------------------------------------------------
# 9. ACCURACY / LOSS PLOT
# ---------------------------------------------------------------
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Baseline CNN Accuracy")
plt.legend(["Train", "Val"])
plt.savefig("Baseline_Accuracy.png")
plt.show()

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Baseline CNN Loss")
plt.legend(["Train", "Val"])
plt.savefig("Baseline_Loss.png")
plt.show()

# ---------------------------------------------------------------
# 10. EVALUATE ON TEST SET
# ---------------------------------------------------------------
y_pred = model.predict(X_test_cnn).argmax(axis=1)

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, target_names=list(emotion_map.values())))

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------
# 11. SAVE MODEL
# ---------------------------------------------------------------
model.save("Baseline_CNN_Model")

print("\n✔ PERSON 1 COMPLETED SUCCESSFULLY.")

spark.stop()
