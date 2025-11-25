# ==============================================================
# PERSON 1 — FULL PIPELINE (LOCAL SPARK)
# FER2013-STYLE FACE EMOTION CLASSIFICATION
# 1) Spark preprocessing (images -> feature vectors)
# 2) Train/Val split + EDA
# 3) Export NPZ for other teammates
# 4) Baseline CNN training + evaluation
# ==============================================================

import os
import io
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
# Local dataset path:
#   /home/sat3812/Final_project/Dataset/train/<class>/*.png
#   /home/sat3812/Final_project/Dataset/test/<class>/*.png
BASE_PATH = "file:///home/sat3812/Final_project/Dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH  = os.path.join(BASE_PATH, "test")

OUTPUT_DIR = "/home/sat3812/Final_project/Output_p1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 48   # FER standard
LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}
NUM_CLASSES = len(LABEL_MAP)

np.random.seed(42)

# -------------------------------------------------------------
# START SPARK (master is provided by spark-submit)
# -------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("FER-Person1-FullPipeline-Local")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.cores", "2")
    .getOrCreate()
)
print("🔥 Spark Session started (local mode).")

# -------------------------------------------------------------
# UDFs — image bytes -> normalized 48×48 vector
# -------------------------------------------------------------
def image_to_vector(bytestr):
    """Convert raw image bytes to flattened normalized grayscale vector."""
    try:
        img = Image.open(io.BytesIO(bytestr)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten().tolist()
    except Exception:
        return None

vec_udf = F.udf(image_to_vector, ArrayType(FloatType()))
label_udf = F.udf(lambda l: LABEL_MAP.get(l, -1), IntegerType())

spark.udf.register("vec_udf", vec_udf)
spark.udf.register("label_udf", label_udf)

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
def load_split(path):
    """Read images from folder and attach integer labels."""
    df = (
        spark.read.format("image")
        .option("dropInvalid", True)
        .load(path)
    )
    # Extract label string from the folder name
    df = df.withColumn(
        "label_str",
        F.regexp_extract(F.col("image.origin"), r".*/([^/]+)/[^/]+$", 1),
    )
    df = df.withColumn("label", label_udf(F.col("label_str")))
    df = df.filter(F.col("label") >= 0)
    return df

print("📂 Loading dataset from:", BASE_PATH)
full_train_df = load_split(TRAIN_PATH)
test_df       = load_split(TEST_PATH)

print("Train images (total):", full_train_df.count())
print("Test images:         ", test_df.count())

# -------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------------------------------------
train_df, val_df = full_train_df.randomSplit([0.8, 0.2], seed=42)
print("After split → Train:", train_df.count(), "| Val:", val_df.count())

# -------------------------------------------------------------
# PREPROCESS & EXPORT NPZ
# -------------------------------------------------------------
def preprocess(df, name):
    """Spark DF -> NumPy arrays -> save as .npz."""
    print(f"⚙️ Preprocessing {name}...")
    df2 = df.withColumn("features", vec_udf(F.col("image.data"))).dropna()
    rows = df2.select("features", "label").collect()

    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int64)

    out_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez_compressed(out_path, X=X, y=y)
    print(f"✔ Saved {name} → {out_path} | shape:", X.shape, y.shape)
    return X, y

X_train, y_train = preprocess(train_df, "train")
X_val,   y_val   = preprocess(val_df,   "val")
X_test,  y_test  = preprocess(test_df,  "test")

# -------------------------------------------------------------
# EDA — CLASS DISTRIBUTION
# -------------------------------------------------------------
def plot_distribution(y, filename="class_distribution.png"):
    counts = Counter(y)
    labels = sorted(counts.keys())
    class_names = [list(LABEL_MAP.keys())[i] for i in labels]
    values = [counts[i] for i in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, values)
    plt.xticks(rotation=45)
    plt.xlabel("Emotion Class")
    plt.ylabel("Count")
    plt.title("Training Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

plot_distribution(y_train)

# -------------------------------------------------------------
# EDA — SAMPLE IMAGES
# -------------------------------------------------------------
def show_samples(X, y, filename="samples.png"):
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    plt.figure(figsize=(6, 6))
    idxs = np.random.choice(len(X), 9, replace=False)
    for i, idx in enumerate(idxs):
        img = X[idx].reshape(IMG_SIZE, IMG_SIZE)
        lab = inv_map[int(y[idx])]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(lab)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

show_samples(X_train, y_train)

spark.stop()
print("🟢 Spark stopped. Preprocessing & EDA complete.")

# =============================================================
# BASELINE CNN TRAINING
# =============================================================
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("\nFinal shapes:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

def build_baseline():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model

model = build_baseline()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "baseline_best.h5"),
        save_best_only=True,
        monitor="val_accuracy",
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=2,
    callbacks=callbacks,
)

# -------------------------------------------------------------
# TRAINING CURVES
# -------------------------------------------------------------
plt.figure()
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Baseline CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "baseline_accuracy.png"))
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Baseline CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "baseline_loss.png"))
plt.close()

# -------------------------------------------------------------
# TEST EVALUATION
# -------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Baseline CNN Test Accuracy = {test_acc:.4f}")

y_pred = model.predict(X_test).argmax(axis=1)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n🚀 PERSON 1 COMPLETED: NPZ + baseline model ready for teammates.")
