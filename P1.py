# ===============================================================
# PERSON 1 — PySpark Image Loader + Preprocessing (VM LOCAL)
# ===============================================================

import os
import numpy as np
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType, StringType
from sklearn.model_selection import train_test_split

# ===============================================================
# 1. START SPARK SESSION
# ===============================================================
spark = SparkSession.builder \
    .appName("FER_PySpark_Image_Loader") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("🔥 Spark session started")

# ===============================================================
# 2. DEFINE DATA PATHS ON YOUR VM
# ===============================================================
train_path = "/home/sat3812/Final_project/Dataset/train"
test_path  = "/home/sat3812/Final_project/Dataset/test"

print("📁 Train Path:", train_path)
print("📁 Test  Path:", test_path)

# ===============================================================
# 3. LOAD IMAGES USING binaryFile WITH RECURSIVE LOOKUP
# ===============================================================
train_df = spark.read.format("binaryFile") \
    .option("recursiveFileLookup", "true") \
    .option("pathGlobFilter", "*.jpg") \
    .load(train_path)

test_df = spark.read.format("binaryFile") \
    .option("recursiveFileLookup", "true") \
    .option("pathGlobFilter", "*.jpg") \
    .load(test_path)

print("Train count:", train_df.count())
print("Test  count:", test_df.count())

# ===============================================================
# 4. EXTRACT CLASS LABEL FROM FOLDER NAME
# ===============================================================
def extract_label_from_path(path):
    return path.split("/")[-2]     # folder before filename

label_udf = udf(extract_label_from_path, StringType())

train_df = train_df.withColumn("label", label_udf(col("path")))
test_df  = test_df.withColumn("label", label_udf(col("path")))

# ===============================================================
# 5. CONVERT IMAGE BINARY → NUMPY ARRAY (48x48 GRAYSCALE)
# ===============================================================
IMG_SIZE = 48

def preprocess_image(content):
    try:
        img = Image.open(BytesIO(content)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img) / 255.0
        return arr.flatten().tolist()
    except:
        return None

from io import BytesIO
pp_udf = udf(preprocess_image, ArrayType(FloatType()))

train_df = train_df.withColumn("pixels", pp_udf(col("content"))).dropna()
test_df  = test_df.withColumn("pixels", pp_udf(col("content"))).dropna()

# ===============================================================
# 6. COLLECT TO DRIVER AS NUMPY ARRAYS
# ===============================================================
train_data = train_df.select("pixels", "label").collect()
test_data  = test_df.select("pixels", "label").collect()

# Convert to NumPy
X_train_full = np.array([row["pixels"] for row in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE)
y_train_full = np.array([row["label"]  for row in train_data])

X_test = np.array([row["pixels"] for row in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE)
y_test = np.array([row["label"]  for row in test_data])

# ===============================================================
# 7. ENCODE LABELS TO NUMBERS
# ===============================================================
classes = sorted(list(set(y_train_full)))
class_to_idx = {c: i for i, c in enumerate(classes)}
print("Class map:", class_to_idx)

y_train_full = np.array([class_to_idx[x] for x in y_train_full])
y_test       = np.array([class_to_idx[x] for x in y_test])

# ===============================================================
# 8. TRAIN/VAL SPLIT
# ===============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

# ===============================================================
# 9. SAVE NPZ OUTPUT FILES
# ===============================================================
np.savez_compressed("train.npz", X=X_train, y=y_train)
np.savez_compressed("val.npz",   X=X_val,   y=y_val)
np.savez_compressed("test.npz",  X=X_test,  y=y_test)

print("✅ Saved train.npz, val.npz, test.npz")
print("🚀 PERSON 1 COMPLETE — Preprocessing Finished.")

spark.stop()
