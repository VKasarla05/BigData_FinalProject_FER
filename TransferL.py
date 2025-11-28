# ==============================================================
# Transfer Learning with MobileNetV2
# ==============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pyspark.sql import SparkSession

# ==============================================================
# 1. SPARK INITIALIZATION 
# ==============================================================

spark = SparkSession.builder \
    .appName("TransferL-MobileNetV2") \
    .master("spark://hadoop1:7077") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark started on:", spark.sparkContext.master)

# ==============================================================
# 2. OUTPUT DIRECTORY
# ==============================================================

output_dir = "/home/sat3812/Final_project/OutputP2"
os.makedirs(output_dir, exist_ok=True)

print("Saving Person2 outputs to:", output_dir)

# ==============================================================
# 3. LOAD NPZ FILES
# ==============================================================

npz_base = "/home/sat3812/Final_project/Dataset/npz"
train = np.load(f"{npz_base}/train.npz")
val   = np.load(f"{npz_base}/val.npz")
test  = np.load(f"{npz_base}/test.npz")

X_train, y_train = train["X"], train["y"]
X_val,   y_val   = val["X"],   val["y"]
X_test,  y_test  = test["X"],  test["y"]

print("Dataset loaded:")
print("Train:", X_train.shape)
print("Val:",   X_val.shape)
print("Test:",  X_test.shape)

# ==============================================================
# 4. PREPROCESSING (Resize + Convert to RGB)
# ==============================================================

IMG_SIZE = 96   # MobileNetV2-compatible

def preprocess(X):
    X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    X = tf.image.resize(X, (IMG_SIZE, IMG_SIZE)).numpy()
    return X

X_train = preprocess(X_train)
X_val   = preprocess(X_val)
X_test  = preprocess(X_test)

print("Final preprocessed shape:", X_train.shape)

# ==============================================================
# 5. BUILD TRANSFER LEARNING MODEL (MobileNetV2)
# ==============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False   # Freeze CNN backbone

# Classification head
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(7, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save model summary
with open(os.path.join(output_dir, "mobilenetv2_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("Model summary saved.")

# ==============================================================
# 6. TRAIN MODEL
# ==============================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=64,
    verbose=2
)

# ==============================================================
# 7. SAVE TRAINING PLOTS
# ==============================================================

plt.figure(figsize=(7,5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("MobileNetV2 Accuracy")
plt.legend()
plt.savefig(f"{output_dir}/accuracy_plot.png")
plt.close()

plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("MobileNetV2 Loss")
plt.legend()
plt.savefig(f"{output_dir}/loss_plot.png")
plt.close()

print("Training plots saved.")

# ==============================================================
# 8. EVALUATE ON TEST SET
# ==============================================================

pred = model.predict(X_test).argmax(axis=1)

report = classification_report(y_test, pred)
cm = confusion_matrix(y_test, pred)

# Save report
with open(f"{output_dir}/classification_report.txt", "w") as f:
    f.write(report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("MobileNetV2 Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()

print("Classification report and confusion matrix saved.")

# ==============================================================
# 9. SAVE TRAINED MODEL
# ==============================================================

model_path = f"{output_dir}/mobilenetv2_person2.h5"
model.save(model_path)

print("Model saved to:", model_path)

# ==============================================================
# 10. STOP SPARK
# ==============================================================

spark.stop()
print("Transfer learning processing complete.")
