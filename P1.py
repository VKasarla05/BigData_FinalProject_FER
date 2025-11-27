# =====================================================================
# PERSON 1 — Spark-Based Preprocessing + Baseline CNN (Happy/Sad/All emotions)
# =====================================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType, FloatType

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# 1. START SPARK SESSION
# =====================================================================

spark = SparkSession.builder \
    .appName("Spark FER Preprocessing") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("\n🔥 Spark started:", spark.sparkContext.master)

# =====================================================================
# 2. DEFINE PATHS ON VM
# =====================================================================

train_path = "/home/sat3812/Final_project/Dataset/train"
test_path  = "/home/sat3812/Final_project/Dataset/test"

output_dir = "/home/sat3812/Final_project/output_p1"
os.makedirs(output_dir, exist_ok=True)

print("\n📁 Train Path:", train_path)
print("📁 Test  Path:", test_path)

# =====================================================================
# 3. LOAD IMAGES USING Spark binaryFile
# =====================================================================

train_df = spark.read.format("binaryFile") \
    .option("recursiveFileLookup", "true") \
    .option("pathGlobFilter", "*.jpg") \
    .load(train_path)

test_df = spark.read.format("binaryFile") \
    .option("recursiveFileLookup", "true") \
    .option("pathGlobFilter", "*.jpg") \
    .load(test_path)

print("\n✔ Train count:", train_df.count())
print("✔ Test count: ", test_df.count())

# =====================================================================
# 4. EXTRACT LABEL FROM PATH
# =====================================================================

def extract_label(path):
    return path.split("/")[-2]

extract_udf = udf(extract_label, StringType())

train_df = train_df.withColumn("label", extract_udf(col("path")))
test_df  = test_df.withColumn("label", extract_udf(col("path")))

# =====================================================================
# 5. CONVERT BINARY → NUMPY (GRAY + RESIZE 48×48)
# =====================================================================

IMG_SIZE = 48

def convert_image(binary_bytes):
    try:
        img = Image.open(tf.io.BytesIO(binary_bytes)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        return np.array(img).flatten().tolist()
    except:
        return None
    
convert_udf = udf(convert_image, ArrayType(FloatType()))

train_df = train_df.withColumn("pixels", convert_udf(col("content")))
test_df  = test_df.withColumn("pixels",  convert_udf(col("content")))

# Drop rows with corrupt images
train_df = train_df.na.drop(subset=["pixels"])
test_df  = test_df.na.drop(subset=["pixels"])

# Collect to driver
train_data = train_df.select("pixels", "label").collect()
test_data  = test_df.select("pixels", "label").collect()

# Convert to NumPy
X_train_full = np.array([np.array(r["pixels"]).reshape(IMG_SIZE, IMG_SIZE) for r in train_data])
y_train_full = np.array([r["label"] for r in train_data])

X_test = np.array([np.array(r["pixels"]).reshape(IMG_SIZE, IMG_SIZE) for r in test_data])
y_test = np.array([r["label"] for r in test_data])

# Encode labels to integers
unique_labels = sorted(list(set(list(y_train_full) + list(y_test))))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_to_label = {i: label for label, i in label_to_int.items()}

y_train_full = np.array([label_to_int[l] for l in y_train_full])
y_test = np.array([label_to_int[l] for l in y_test])

num_classes = len(unique_labels)

print("\nUnique labels:", unique_labels)
print("Number of classes:", num_classes)

# =====================================================================
# 6. SPLIT 75% / 15% / 15%
# =====================================================================

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_train_full, y_train_full,
    test_size=0.30, random_state=42, stratify=y_train_full
)

X_val, X_test_final, y_val, y_test_final = train_test_split(
    X_tmp, y_tmp,
    test_size=0.50, random_state=42, stratify=y_tmp
)

print("\n📊 Dataset Split Shapes:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test_final.shape)

# =====================================================================
# 7. SAVE NPZ FILES FOR NEXT TEAM MEMBERS
# =====================================================================

np.savez(os.path.join(output_dir, "train.npz"), X=X_train, y=y_train)
np.savez(os.path.join(output_dir, "val.npz"),   X=X_val,   y=y_val)
np.savez(os.path.join(output_dir, "test.npz"),  X=X_test_final, y=y_test_final)

print("\n💾 Saved NPZ files to:", output_dir)

# =====================================================================
# 8. VISUALIZATIONS (REQUIRED FOR SLIDES)
# =====================================================================

# ---- Class Distribution ----
counts = Counter(y_train)
plt.figure()
plt.bar([int_to_label[c] for c in counts.keys()], counts.values())
plt.title("Class Distribution (Training Set)")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

# ---- Sample Images ----
plt.figure(figsize=(6,6))
indices = np.random.choice(len(X_train), 9)
for i, idx in enumerate(indices):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[idx], cmap="gray")
    plt.title(int_to_label[y_train[idx]])
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sample_images.png"))
plt.close()

print("📊 Saved EDA visualizations.")

# =====================================================================
# 9. BASELINE CNN
# =====================================================================

X_train_cnn = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val_cnn   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test_cnn  = X_test_final.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def build_baseline_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = build_baseline_cnn()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Save summary
with open(os.path.join(output_dir, "cnn_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("\n🧠 Training Baseline CNN...")
history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=10,
    batch_size=32,
    verbose=2
)

# =====================================================================
# 10. TRAINING CURVES
# =====================================================================

plt.figure()
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("CNN Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, "cnn_accuracy.png"))
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("CNN Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "cnn_loss.png"))
plt.close()

# =====================================================================
# 11. TEST EVALUATION
# =====================================================================

preds = model.predict(X_test_cnn).argmax(axis=1)

print("\n📌 Classification Report:")
print(classification_report(y_test_final, preds, target_names=unique_labels))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test_final, preds))

print("\n🎉 PERSON 1 COMPLETE — Data Ready + Baseline Model Trained.")
spark.stop()
