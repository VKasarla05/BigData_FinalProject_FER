# ==============================================================
# PERSON 1 — PySpark Preprocessing + EDA + Export + Baseline CNN
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
# CONFIGURE LOCAL DATA PATH (YOUR VM PATH)
# -------------------------------------------------------------
BASE_PATH = "/home/sat3812/Final_project/Dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH  = os.path.join(BASE_PATH, "test")
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
NUM_CLASSES = len(LABEL_MAP)

# -------------------------------------------------------------
# START SPARK (LOCAL VM ONLY)
# -------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("FER-Person1")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
print("🔥 Spark Started!")

# -------------------------------------------------------------
# IMAGE → VECTOR UDF
# -------------------------------------------------------------
def img_to_vec(bytestr):
    try:
        img = Image.open(io.BytesIO(bytestr)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten().tolist()
    except:
        return None

vec_udf = F.udf(lambda b: img_to_vec(b), ArrayType(FloatType()))

label_udf = F.udf(lambda l: LABEL_MAP.get(l, -1), IntegerType())

# -------------------------------------------------------------
# LOAD IMAGES FROM LOCAL FOLDERS (Spark supports this!)
# -------------------------------------------------------------
def load_split(path):
    df = (
        spark.read.format("image")
        .option("dropInvalid", True)
        .load(path)
    )

    df = df.withColumn(
        "label_str",
        F.regexp_extract(F.col("image.origin"), r".*/([^/]+)/[^/]+$", 1)
    )
    df = df.withColumn("label", label_udf("label_str"))
    return df.filter("label >= 0")

print("📂 Loading data...")
train_df = load_split(TRAIN_PATH)
test_df  = load_split(TEST_PATH)

print("Train count:", train_df.count())
print("Test count:",  test_df.count())

# -------------------------------------------------------------
# TRAIN/VAL SPLIT
# -------------------------------------------------------------
train_df, val_df = train_df.randomSplit([0.8, 0.2], seed=42)

# -------------------------------------------------------------
# PREPROCESS: Convert Spark → NumPy
# -------------------------------------------------------------
def preprocess(df, name):
    print(f"⚙️ Processing {name}...")
    df2 = df.withColumn("features", vec_udf("image.data")).dropna()

    rows = df2.select("features", "label").collect()
    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int64)

    np.savez_compressed(os.path.join(OUTPUT_DIR, f"{name}.npz"), X=X, y=y)
    print(f"✔ Saved {name}.npz → {X.shape}")
    return X, y

X_train, y_train = preprocess(train_df, "train")
X_val, y_val     = preprocess(val_df, "val")
X_test, y_test   = preprocess(test_df, "test")

spark.stop()
print("🟢 Spark stopped.")

# -------------------------------------------------------------
# EDA — CLASS DISTRIBUTION
# -------------------------------------------------------------
def plot_dist(y, title):
    counts = Counter(y)
    names = list(LABEL_MAP.keys())
    vals = [counts.get(LABEL_MAP[n], 0) for n in names]

    plt.figure(figsize=(8,4))
    plt.bar(names, vals)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{title}.png")
    plt.show()

plot_dist(y_train, "Train_Class_Distribution")

# -------------------------------------------------------------
# EDA — SAMPLE IMAGES
# -------------------------------------------------------------
def show_samples(X, y):
    inv = {v:k for k,v in LABEL_MAP.items()}
    idxs = np.random.choice(len(X), 9, replace=False)
    plt.figure(figsize=(6,6))
    for i, idx in enumerate(idxs):
        plt.subplot(3,3,i+1)
        plt.imshow(X[idx].reshape(48,48), cmap="gray")
        plt.title(inv[y[idx]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_samples(X_train, y_train)

# -------------------------------------------------------------
# CNN INPUT SHAPES
# -------------------------------------------------------------
X_train_cnn = X_train.reshape(-1, 48, 48, 1)
X_val_cnn   = X_val.reshape(-1, 48, 48, 1)
X_test_cnn  = X_test.reshape(-1, 48, 48, 1)

# -------------------------------------------------------------
# BASELINE CNN
# -------------------------------------------------------------
def build_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same",
                               input_shape=(48,48,1)),
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

model = build_cnn()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("🚀 Training CNN...")
history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=15,
    batch_size=32,
    verbose=2
)

# -------------------------------------------------------------
# TEST SCORE
# -------------------------------------------------------------
loss, acc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {acc:.4f}")

y_pred = model.predict(X_test_cnn).argmax(axis=1)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n🚀 PERSON 1 DONE — Data ready for Person 2 & 3.")
