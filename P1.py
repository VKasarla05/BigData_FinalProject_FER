# =============================================================
# PERSON 1 — SPARK + PREPROCESSING + EDA + BASELINE CNN
# Final full implementation
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from sklearn.model_selection import train_test_split
import io

# =============================================================
# CONFIGURATION
# =============================================================
DATASET_PATH = "/home/sat3812/Final_project/Dataset"
OUTPUT_DIR = "/home/sat3812/Final_project/output_p1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 48

LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

INV_LABEL = {v: k for k, v in LABEL_MAP.items()}

print("📂 Dataset Path:", DATASET_PATH)
print("💾 Output Path:", OUTPUT_DIR)

# =============================================================
# START SPARK SESSION
# =============================================================
spark = (
    SparkSession.builder
    .appName("FER-Spark-Preprocessing")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)
print("🚀 Spark Started:", spark)

# =============================================================
# UDF — Convert image bytes → vector
# =============================================================
def image_to_vector(bytestr):
    try:
        img = Image.open(io.BytesIO(bytestr)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten().tolist()
    except:
        return None

vec_udf = F.udf(lambda b: image_to_vector(b), ArrayType(FloatType()))
label_udf = F.udf(lambda l: LABEL_MAP.get(l, -1), IntegerType())

# =============================================================
# LOAD DATA USING SPARK
# =============================================================
def load_spark_split(split):
    path = os.path.join(DATASET_PATH, split)
    df = (
        spark.read.format("image")
        .option("dropInvalid", True)
        .load(path)
    )
    df = df.withColumn("label_str", F.regexp_extract(F.col("image.origin"), r".*/([^/]+)/[^/]+$", 1))
    df = df.withColumn("label", label_udf("label_str"))
    df = df.filter(F.col("label") >= 0)
    return df

print("📥 Loading Train/Test using Spark...")
train_df = load_spark_split("train")
test_df = load_spark_split("test")

print("🟦 Train Count:", train_df.count())
print("🟧 Test Count:", test_df.count())

# =============================================================
# EXTRACT FEATURES FROM SPARK DF
# =============================================================
def extract_numpy(df, name):
    print(f"\n⚙️ Processing {name}...")
    df = df.withColumn("features", vec_udf("image.data")).dropna()
    rows = df.select("features", "label").collect()
    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int64)

    output_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    np.savez_compressed(output_path, X=X, y=y)
    print(f"✔ Saved {name}.npz → {output_path}  Shape: {X.shape}")

    return X, y

X_train_full, y_train_full = extract_numpy(train_df, "train_full")
X_test, y_test = extract_numpy(test_df, "test")

spark.stop()
print("\n🟢 Spark stopped — preprocessing completed!")

# =============================================================
# TRAIN / VAL / TEST SPLIT 75 / 15 / 15
# =============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.30,         # 15% val + 15% test
    stratify=y_train_full,
    random_state=42
)

# Split val/test equally
X_val, X_extra_test, y_val, y_extra_test = train_test_split(
    X_val, y_val,
    test_size=0.50,
    stratify=y_val,
    random_state=42
)

# Append extra test to main test
X_test = np.concatenate([X_test, X_extra_test])
y_test = np.concatenate([y_test, y_extra_test])

# =============================================================
# SAVE FINAL SPLITS
# =============================================================
np.savez_compressed(os.path.join(OUTPUT_DIR, "train.npz"), X=X_train, y=y_train)
np.savez_compressed(os.path.join(OUTPUT_DIR, "val.npz"),   X=X_val,   y=y_val)
np.savez_compressed(os.path.join(OUTPUT_DIR, "test.npz"),  X=X_test,  y=y_test)

print("\n💾 Saved train/val/test NPZ files successfully!")

# =============================================================
# EDA — CLASS DISTRIBUTION
# =============================================================
def plot_distribution(y, title, filename):
    counts = Counter(y)
    labels = [INV_LABEL[i] for i in sorted(counts.keys())]
    values = [counts[i] for i in sorted(counts.keys())]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved plot → {save_path}")

plot_distribution(y_train, "Training Class Distribution", "train_distribution.png")

# =============================================================
# EDA — SAMPLE IMAGES
# =============================================================
def show_samples(X, y, filename):
    plt.figure(figsize=(6,6))
    idxs = np.random.choice(len(X), 9, replace=False)

    for i, idx in enumerate(idxs):
        img = X[idx].reshape(IMG_SIZE, IMG_SIZE)
        plt.subplot(3,3,i+1)
        plt.imshow(img, cmap="gray")
        plt.title(INV_LABEL[y[idx]])
        plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📸 Saved sample images → {save_path}")

show_samples(X_train, y_train, "sample_grid.png")

# =============================================================
# BASELINE CNN MODEL
# =============================================================
X_train_cnn = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val_cnn   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test_cnn  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

NUM_CLASSES = len(LABEL_MAP)

def build_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
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

model = build_cnn()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# =============================================================
# PRINT & SAVE CNN SUMMARY
# =============================================================

summary_path = os.path.join(OUTPUT_DIR, "cnn_summary.txt")

with open(summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("\n🧱 CNN Model Summary:")
model.summary()

print(f"\n📄 CNN Summary saved to: {summary_path}")


print("\n🚀 Training Baseline CNN...")
history = model.fit(X_train_cnn, y_train, validation_data=(X_val_cnn, y_val), epochs=15, batch_size=32)

# =============================================================
# SAVE TRAINING PLOTS
# =============================================================
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Baseline CNN Accuracy")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "cnn_accuracy.png"))
plt.close()

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Baseline CNN Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "cnn_loss.png"))
plt.close()

# =============================================================
# EVALUATE BASELINE CNN
# =============================================================
test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\n🎯 Baseline CNN Test Accuracy = {test_acc:.4f}")

y_pred = model.predict(X_test_cnn).argmax(axis=1)

print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔢 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n🚀 PERSON 1 COMPLETE — Data ready for Person 2 & Person 3!")
