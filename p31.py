# ================================
# Person3 – Fine-tune Person2 model
# ================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession

# -------------------------------
# 0. Spark session (for requirement)
# -------------------------------
spark = SparkSession.builder \
    .appName("Person3-Finetuning") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark session started on:", spark.sparkContext.master)

# -------------------------------
# 1. Paths
# -------------------------------
BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
OUTPUT = f"{BASE}/Output_P3"
MODEL_FROM_P2 = f"{BASE}/Output_2/mobilenetv2_person2.h5"  # change if your file name differs

os.makedirs(OUTPUT, exist_ok=True)
print("Saving Person3 outputs to:", OUTPUT)

# -------------------------------
# 2. Load NPZ files (data already pre-split)
# -------------------------------
train_npz = np.load(os.path.join(NPZ_PATH, "train.npz"))
val_npz   = np.load(os.path.join(NPZ_PATH, "val.npz"))
test_npz  = np.load(os.path.join(NPZ_PATH, "test.npz"))

X_train, y_train = train_npz["X"], train_npz["y"]
X_val,   y_val   = val_npz["X"],   val_npz["y"]
X_test,  y_test  = test_npz["X"],  test_npz["y"]

print("Train:", X_train.shape, "| Labels:", len(y_train))
print("Val:  ", X_val.shape,   "| Labels:", len(y_val))
print("Test: ", X_test.shape,  "| Labels:", len(y_test))

# We do TF training on the driver only
spark.stop()
print("Spark session stopped after data loading.")

# -------------------------------
# 3. Preprocessing for MobileNetV2
# -------------------------------
IMG_SIZE = 96

def preprocess(x):
    x = np.repeat(x[..., np.newaxis], 3, axis=-1)    # (N, 48, 48) -> (N, 48, 48, 3)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x.astype("float32")

X_train = preprocess(X_train)
X_val   = preprocess(X_val)
X_test  = preprocess(X_test)

print("After preprocessing:", X_train.shape)

# -------------------------------
# 4. Load Person2 model
# -------------------------------
print("Loading Person2 MobileNetV2 model...")
p2_model = tf.keras.models.load_model(MODEL_FROM_P2)
print("Person2 model loaded.")

with open(os.path.join(OUTPUT, "person2_model_summary_before_p3.txt"), "w") as f:
    p2_model.summary(print_fn=lambda s: f.write(s + "\n"))

# -------------------------------
# 5. Attach custom conv block to last 4D feature map
# -------------------------------

# Find last layer with 4D output (H, W, C) – i.e., last conv feature map
last_conv_layer = None
for layer in reversed(p2_model.layers):
    # some layers don't have .output_shape; use getattr safely
    out_shape = getattr(layer, "output_shape", None)
    if out_shape is not None and len(out_shape) == 4:
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise ValueError("Could not find a 4D conv layer in Person2 model.")

print("Using last conv layer for custom block:", last_conv_layer.name)

conv_output = last_conv_layer.output  # shape (None, H, W, C)

x = tf.keras.layers.Conv2D(
    128, (3, 3),
    padding="same",
    activation="relu",
    name="custom_conv_p3"
)(conv_output)

x = tf.keras.layers.GlobalAveragePooling2D(name="custom_gap_p3")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(7, activation="softmax")(x)

model = tf.keras.Model(inputs=p2_model.input, outputs=outputs)

# -------------------------------
# 6. Fine-tune last 20 layers
# -------------------------------
for layer in model.layers[:-20]:
    layer.trainable = False
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

with open(os.path.join(OUTPUT, "person3_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))

# -------------------------------
# 7. Training
# -------------------------------
print("Starting Person3 fine-tuning...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    verbose=2
)

# -------------------------------
# 8. Save training curves
# -------------------------------
plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Person3 Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "accuracy_p3.png"))
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Person3 Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "loss_p3.png"))
plt.close()

print("Saved training plots.")

# -------------------------------
# 9. Evaluation on test set
# -------------------------------
print("Evaluating Person3 model on test set...")
preds = model.predict(X_test, verbose=0).argmax(axis=1)

report = classification_report(y_test, preds)
with open(os.path.join(OUTPUT, "p3_classification_report.txt"), "w") as f:
    f.write(report)

print(report)

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Person3 Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "confusion_matrix_p3.png"))
plt.close()

# -------------------------------
# 10. Save final model
# -------------------------------
final_model_path = os.path.join(OUTPUT, "mobilenetv2_person3_finetuned.h5")
model.save(final_model_path)
print("Saved Person3 fine-tuned model to:", final_model_path)
