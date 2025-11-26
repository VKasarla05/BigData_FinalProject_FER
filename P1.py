# ==============================================================
# PERSON 1 — COMPLETE UPDATED PIPELINE WITH JPG/JPEG SUPPORT
# FER2013-STYLE FACE EMOTION CLASSIFICATION
# ==============================================================

import os
import io
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------------
# FIXED PATHS — DIRECT GLOB PATHS (NO os.path.join)
# -------------------------------------------------------------
TRAIN_DIR = "file:///home/sat3812/Final_project/Dataset/train"
TEST_DIR  = "file:///home/sat3812/Final_project/Dataset/test"

# Spark needs file globs to load all file types recursively
TRAIN_PATH = TRAIN_DIR + "/*/*"
TEST_PATH  = TEST_DIR  + "/*/*"

OUTPUT_DIR = "/home/sat3812/Final_project/Output_p1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 48

LABEL_MAP = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
    "neutral": 4, "sad": 5, "surprise": 6
}
NUM_CLASSES = len(LABEL_MAP)

# -------------------------------------------------------------
# START SPARK (LOCAL MODE)
# -------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("FER-Person1-FullPipeline-LOCAL")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.executor.cores", "2")
    .getOrCreate()
)
print("🔥 Spark session started in LOCAL mode")

# -------------------------------------------------------------
# UDF: Convert image bytes → normalized 48×48 vector
# -------------------------------------------------------------
def image_to_vector(bytestr):
    try:
        img = Image.open(io.BytesIO(bytestr)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten().tolist()
    except Exception:
        return None

vec_udf = F.udf(image_to_vector, ArrayType(FloatType()))
label_udf = F.udf(lambda s: LABEL_MAP.get(s, -1), IntegerType())

spark.udf.register("vec_udf", vec_udf)
spark.udf.register("label_udf", label_udf)

# -------------------------------------------------------------
# LOAD DATA (Supports JPG/JPEG/PNG and UPPERCASE)
# -------------------------------------------------------------
def load_split(path):
    df = (
        spark.read.format("image")
        .option("dropInvalid", True)
        .load(path)
    )

    # Case-insensitive extension detection
    df = df.withColumn(
        "label_str",
        F.regexp_extract(
            F.col("image.origin"),
            r"/([^/]+)/[^/]+\.(?i:jpg|jpeg|png)$",
            1
        )
    )

    df = df.withColumn("label", label_udf(F.col("label_str")))
    df = df.filter(F.col("label") >= 0)
    return df

print(f"📂 Loading dataset from:")
print(TRAIN_PATH)
print(TEST_PATH)

full_train_df = load_split(TRAIN_PATH)
test_df       = load_split(TEST_PATH)

print("Train images:", full_train_df.count())
print("Test images:",  test_df.count())

# -------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------------------------------------
train_df, val_df = full_train_df.randomSplit([0.8, 0.2], seed=42)
print("After split → Train:", train_df.count(), "| Val:", val_df.count())

# -------------------------------------------------------------
# PREPROCESS (Spark DF → NumPy → NPZ)
# -------------------------------------------------------------
def preprocess(df, name):
    print(f"⚙️ Preprocessing {name}...")

    df2 = df.withColumn("features", vec_udf(F.col("image.data"))).dropna()
    rows = df2.select("features", "label").collect()

    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int64)

    np.savez_compressed(f"{OUTPUT_DIR}/{name}.npz", X=X, y=y)
    print(f"✔ Saved {name}.npz | shape:", X.shape, y.shape)
    return X, y

X_train, y_train = preprocess(train_df, "train")
X_val,   y_val   = preprocess(val_df,   "val")
X_test,  y_test  = preprocess(test_df,  "test")

# -------------------------------------------------------------
# EDA — CLASS DISTRIBUTION
# -------------------------------------------------------------
def plot_distribution(y):
    counts = Counter(y)
    labels = sorted(counts.keys())
    class_names = [list(LABEL_MAP.keys())[i] for i in labels]
    values = [counts[l] for l in labels]

    plt.figure(figsize=(6,4))
    plt.bar(class_names, values)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
    plt.close()

plot_distribution(y_train)

# -------------------------------------------------------------
# EDA — SAMPLE IMAGES
# -------------------------------------------------------------
def show_samples(X, y):
    inv = {v:k for k,v in LABEL_MAP.items()}
    plt.figure(figsize=(6,6))
    idxs = np.random.choice(len(X), 9, replace=False)
    for i, idx in enumerate(idxs):
        img = X[idx].reshape(IMG_SIZE, IMG_SIZE)
        plt.subplot(3,3,i+1)
        plt.imshow(img, cmap="gray")
        plt.title(inv[int(y[idx])])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sample_images.png")
    plt.close()

show_samples(X_train, y_train)

spark.stop()
print("🟢 Spark preprocessing complete!")

# =============================================================
# BASELINE CNN TRAINING
# =============================================================
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def build_baseline():
    return tf.keras.Sequential([
        tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 1)),

        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])

model = build_baseline()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(f"{OUTPUT_DIR}/baseline_best.h5", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=2,
    callbacks=callbacks
)

# -------------------------------------------------------------
# TRAINING CURVES
# -------------------------------------------------------------
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/baseline_accuracy.png")
plt.close()

plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/baseline_loss.png")
plt.close()

# -------------------------------------------------------------
# TEST EVALUATION
# -------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Baseline CNN Test Accuracy = {test_acc:.4f}")

y_pred = model.predict(X_test).argmax(axis=1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n🚀 PERSON 1 COMPLETE — NPZ + baseline model ready for teammates.")
