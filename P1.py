# ==============================================================
# PERSON 1 — FINAL PIPELINE: FER2013 FACE EMOTION CLASSIFICATION
# 1. Spark-based preprocessing (image → feature vectors)
# 2. Train/Validation split + EDA visualization
# 3. Export to NPZ format for reuse
# 4. Baseline CNN training + evaluation
# ==============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from PIL import Image
import io
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------------
# CONFIGURATION — data paths and constants
# -------------------------------------------------------------
# Change BASE_PATH according to your environment (HDFS or local)
BASE_PATH = "file:///home/sat3812/Final_project/Dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH  = os.path.join(BASE_PATH, "test")
OUTPUT_DIR = "/home/sat3812/Final_project/Output_p1/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_SIZE = 48   # Standard FER image size

# Map emotion names to integer labels (0–6)
LABEL_MAP = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
    "neutral": 4, "sad": 5, "surprise": 6
}
NUM_CLASSES = len(LABEL_MAP)

# -------------------------------------------------------------
# INITIALIZE SPARK SESSION
# -------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("FER-Person1-FullPipeline")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.cores", "2")
    .getOrCreate()
)
print("🔥 Spark Session started.")

# -------------------------------------------------------------
# UDFs — convert image bytes to normalized pixel vector
# -------------------------------------------------------------
def image_to_vector(bytestr):
    """Convert raw image bytes to a flattened normalized (48×48) grayscale vector."""
    try:
        img = Image.open(io.BytesIO(bytestr)).convert("L")  # Convert to grayscale
        img = img.resize((IMG_SIZE, IMG_SIZE))              # Standardize size
        arr = np.asarray(img, dtype=np.float32) / 255.0     # Normalize pixels
        return arr.flatten().tolist()
    except:
        return None

# Register Spark UDFs for image and label conversion
vec_udf = F.udf(lambda b: image_to_vector(b), ArrayType(FloatType()))
spark.udf.register("vec_udf", vec_udf)
label_udf = F.udf(lambda l: LABEL_MAP.get(l, -1), IntegerType())
spark.udf.register("label_udf", label_udf)

# -------------------------------------------------------------
# LOAD DATA FROM FOLDERS INTO SPARK DATAFRAME
# -------------------------------------------------------------
def load_split(path):
    """Read images and extract labels from folder names."""
    df = (
        spark.read.format("image")
        .option("dropInvalid", True)
        .load(path)
    )
    # Extract label name from file path and map to integers
    df = df.withColumn("label_str", F.regexp_extract(F.col("image.origin"), r".*/([^/]+)/[^/]+$", 1))
    df = df.withColumn("label", label_udf(F.col("label_str")))
    df = df.filter(F.col("label") >= 0)
    return df

print("📂 Loading dataset...")
full_train_df = load_split(TRAIN_PATH)
test_df = load_split(TEST_PATH)
print("Train images:", full_train_df.count())
print("Test images:",  test_df.count())

# -------------------------------------------------------------
# SPLIT TRAINING SET → TRAIN & VALIDATION
# -------------------------------------------------------------
train_df, val_df = full_train_df.randomSplit([0.8, 0.2], seed=42)
print("After split → Train:", train_df.count(), "| Val:", val_df.count())

# -------------------------------------------------------------
# PREPROCESS DATA AND EXPORT AS COMPRESSED NPZ FILES
# -------------------------------------------------------------
def preprocess(df, name):
    """Convert Spark DF → normalized NumPy arrays and save to disk."""
    print(f"⚙️ Preprocessing {name}...")
    df2 = df.withColumn("features", vec_udf(F.col("image.data"))).dropna()
    rows = df2.select("features", "label").collect()

    # Convert to NumPy arrays
    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int64)

    # Save compressed arrays for reuse
    out = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez_compressed(out, X=X, y=y)
    print(f"✔ Saved {name} →", out, "|", X.shape, y.shape)
    return X, y

X_train, y_train = preprocess(train_df, "train")
X_val, y_val = preprocess(val_df, "val")
X_test, y_test = preprocess(test_df, "test")

# -------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS — CLASS DISTRIBUTION
# -------------------------------------------------------------
def plot_distribution(y):
    """Plot number of samples per emotion class."""
    counts = Counter(y)
    labels = sorted(counts.keys())
    names = [list(LABEL_MAP.keys())[l] for l in labels]
    values = [counts[l] for l in labels]

    plt.figure(figsize=(6,4))
    plt.bar(names, values)
    plt.xticks(rotation=45)
    plt.title("Training Class Distribution")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.show()

plot_distribution(y_train)

# -------------------------------------------------------------
# EDA — DISPLAY RANDOM SAMPLE IMAGES
# -------------------------------------------------------------
def show_samples(X, y, filename="samples.png"):
    """Visualize random training samples by emotion label."""
    plt.figure(figsize=(6,6))
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    idxs = np.random.choice(len(X), 9, replace=False)
    for i, idx in enumerate(idxs):
        img = X[idx].reshape(IMG_SIZE, IMG_SIZE)
        lab = inv_map[y[idx]]
        plt.subplot(3,3,i+1)
        plt.imshow(img, cmap="gray")
        plt.title(lab)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

show_samples(X_train, y_train)
spark.stop()
print("🟢 Spark stopped. Preprocessing & EDA complete.")

# -------------------------------------------------------------
# BASELINE CNN TRAINING
# -------------------------------------------------------------
# Reshape flattened vectors → image tensors for CNN input
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("\nFinal shapes:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# -------------------------------------------------------------
# DEFINE BASELINE CNN MODEL
# -------------------------------------------------------------
def build_baseline():
    """Simple CNN model with 3 convolutional blocks and a dense layer."""
    model = tf.keras.Sequential([
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
    return model

model = build_baseline()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------------------
# TRAIN BASELINE MODEL WITH EARLY STOPPING AND CHECKPOINTS
# -------------------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("baseline_best.h5", save_best_only=True)
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
# TRAINING PERFORMANCE PLOTS
# -------------------------------------------------------------
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Baseline CNN Accuracy")
plt.legend()
plt.savefig("baseline_accuracy.png")
plt.show()

plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Baseline CNN Loss")
plt.legend()
plt.savefig("baseline_loss.png")
plt.show()

# -------------------------------------------------------------
# EVALUATE ON TEST DATA
# -------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Baseline CNN Test Accuracy = {test_acc:.4f}")

# Generate predictions and reports
y_pred = model.predict(X_test).argmax(axis=1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🚀 PERSON 1 COMPLETED: Data ready for next teammates.")
