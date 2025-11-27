# ===============================================================
# PERSON 1 — SPARK IMAGE INGESTION + PREPROCESSING + EDA + BASELINE CNN
# ===============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession
import tensorflow as tf
from PIL import Image
from collections import Counter

# ===============================================================
# PATHS — UPDATE ACCORDING TO YOUR VM
# ===============================================================

DATASET_PATH = "/home/sat3812/Final_project/Dataset"
OUTPUT_DIR = "/home/sat3812/Final_project/output_p1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(DATASET_PATH, "train")
test_path  = os.path.join(DATASET_PATH, "test")

print("📂 Train Path:", train_path)
print("📂 Test  Path:", test_path)

# ===============================================================
# START SPARK SESSION
# ===============================================================

spark = (
    SparkSession.builder
        .appName("Spark FER Image Loader")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
)

print("\n✔ Spark started.\n")

# ===============================================================
# LOAD IMAGES USING SPARK binaryFile
# ===============================================================
# VERY IMPORTANT → case-insensitive filter "*.[jJ][pP][gG]"

train_df = (
    spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.[jJ][pP][gG]")
        .load(train_path)
)

test_df = (
    spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.[jJ][pP][gG]")
        .load(test_path)
)

print("Train count:", train_df.count())
print("Test count: ", test_df.count())

# ===============================================================
# EXTRACT CLASS LABEL FROM FILE PATH
# ===============================================================

def extract_label(path):
    return path.split("/")[-2]

extract_label_udf = spark.udf.register("extractLabel", extract_label)

train_df = train_df.withColumn("label", extract_label_udf(train_df["path"]))
test_df  = test_df.withColumn("label", extract_label_udf(test_df["path"]))

# ===============================================================
# COLLECT LABELS & UNIQUE CLASSES
# ===============================================================

labels_train = [row["label"] for row in train_df.select("label").collect()]
labels_test  = [row["label"] for row in test_df.select("label").collect()]

classes = sorted(list(set(labels_train)))
print("\nUnique labels:", classes)
print("Number of classes:", len(classes))

class_to_idx = {cls: i for i, cls in enumerate(classes)}
print("\nClass index map:", class_to_idx)

# ===============================================================
# CONVERT SPARK DF → NUMPY ARRAYS
# ===============================================================

IMG_SIZE = 48

def spark_to_numpy(df):
    X = []
    y = []

    rows = df.select("path", "label", "content").collect()

    for row in rows:
        try:
            img = Image.open(
                tf.io.BytesIO(row["content"])
            ).convert("L")

            img = img.resize((IMG_SIZE, IMG_SIZE))
            X.append(np.array(img) / 255.0)
            y.append(class_to_idx[row["label"]])

        except:
            print("⚠️ Skipped bad file:", row["path"])
            continue

    return np.array(X), np.array(y)

print("\n📥 Converting train images…")
X_train_full, y_train_full = spark_to_numpy(train_df)

print("📥 Converting test images…")
X_test, y_test = spark_to_numpy(test_df)

print("\nTrain images:", X_train_full.shape)
print("Test images:", X_test.shape)

# ===============================================================
# TRAIN/VAL SPLIT → 75/15/15
# ===============================================================

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_train_full, y_train_full, test_size=0.30, random_state=42, stratify=y_train_full
)

X_val, X_unused, y_val, y_unused = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

print("\nFinal splits:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# ===============================================================
# SAVE NPZ FILES FOR NEXT PERSONS
# ===============================================================

np.savez_compressed(os.path.join(OUTPUT_DIR, "train.npz"), X=X_train, y=y_train)
np.savez_compressed(os.path.join(OUTPUT_DIR, "val.npz"),   X=X_val,   y=y_val)
np.savez_compressed(os.path.join(OUTPUT_DIR, "test.npz"),  X=X_test,  y=y_test)

print(f"\n✔ NPZ files saved to: {OUTPUT_DIR}\n")

# ===============================================================
# EDA - CLASS DISTRIBUTION
# ===============================================================

def plot_distribution(labels, title):
    counter = Counter(labels)
    classes = list(class_to_idx.keys())
    values  = [counter.get(class_to_idx[c], 0) for c in classes]

    plt.figure(figsize=(8, 4))
    plt.bar(classes, values, color="skyblue")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

plot_distribution(y_train, "Training Set Class Distribution")
plot_distribution(y_val,   "Validation Set Class Distribution")
plot_distribution(y_test,  "Test Set Class Distribution")

# ===============================================================
# SAMPLE IMAGES VISUALIZATION
# ===============================================================

def show_samples(X, y, title):
    plt.figure(figsize=(6, 6))
    indices = np.random.choice(len(X), 9, replace=False)
    inv_map = {v: k for k, v in class_to_idx.items()}

    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i+1)
        plt.imshow(X[idx], cmap="gray")
        plt.title(inv_map[y[idx]])
        plt.axis("off")

    plt.suptitle(title)
    plt.show()

show_samples(X_train, y_train, "Sample Training Images")

# ===============================================================
# BASELINE CNN MODEL (for Stage 1)
# ===============================================================

X_train_cnn = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val_cnn   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test_cnn  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(len(classes), activation="softmax")
])

print("\n📄 CNN Model Summary:\n")
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train_cnn, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_cnn, y_val),
    verbose=2
)

# ===============================================================
# PLOTS
# ===============================================================

plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("CNN Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("CNN Loss")
plt.legend()
plt.show()

# ===============================================================
# TEST PERFORMANCE
# ===============================================================

preds = model.predict(X_test_cnn).argmax(axis=1)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, preds))

print("\n📊 Confusion Matrix:\n")
print(confusion_matrix(y_test, preds))

print("\n🎉 PERSON 1 COMPLETE — Data ready for Person 2, 3, 4.\n")

spark.stop()
