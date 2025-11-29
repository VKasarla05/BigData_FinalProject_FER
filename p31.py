import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession

# ===============================================================
# 0. Spark session
# ===============================================================
spark = SparkSession.builder \
    .appName("Person3-Finetuning") \
    .master("spark://192.168.13.134:7077") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark session started")

# ===============================================================
# 1. Paths
# ===============================================================
BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
OUTPUT = f"{BASE}/Output_P3"
MODEL_FROM_P2 = f"{BASE}/Output_2/mobilenetv2_person2.h5"

os.makedirs(OUTPUT, exist_ok=True)

print("Saving Person3 outputs to:", OUTPUT)

# ===============================================================
# 2. Load NPZ files (Spark used for multi-VM requirement)
# ===============================================================
train_npz = np.load(f"{NPZ_PATH}/train.npz")
val_npz   = np.load(f"{NPZ_PATH}/val.npz")
test_npz  = np.load(f"{NPZ_PATH}/test.npz")

X_train, y_train = train_npz["X"], train_npz["y"]
X_val,   y_val   = val_npz["X"],   val_npz["y"]
X_test,  y_test  = test_npz["X"],  test_npz["y"]

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

spark.stop()
print("Spark session closed.")

# ===============================================================
# 3. Preprocessing
# ===============================================================
IMG_SIZE = 96

def preprocess(x):
    x = np.repeat(x[..., np.newaxis], 3, axis=-1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x.astype("float32")

X_train = preprocess(X_train)
X_val   = preprocess(X_val)
X_test  = preprocess(X_test)

print("Processed:", X_train.shape)

# ===============================================================
# 4. Load PERSON2 MODEL
# ===============================================================
print("Loading Person2 MobileNetV2...")
p2_model = tf.keras.models.load_model(MODEL_FROM_P2)
print("Loaded successfully.")

# Save summary
with open(f"{OUTPUT}/p2_model_summary_before_p3.txt","w") as f:
    p2_model.summary(print_fn=lambda s: f.write(s+"\n"))

# ===============================================================
# 5. ADD CUSTOM CONV BLOCK BEFORE GAP
# ===============================================================
last_layer = p2_model.layers[-2].output       # GAP input

x = tf.keras.layers.Conv2D(
    128, (3,3), padding="same", activation="relu", name="custom_conv_p3"
)(last_layer)

x = tf.keras.layers.GlobalAveragePooling2D(name="custom_gap_p3")(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(7, activation="softmax")(x)

model = tf.keras.Model(inputs=p2_model.input, outputs=outputs)

# Unfreeze last 20 layers for fine-tuning
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save new summary
with open(f"{OUTPUT}/person3_model_summary.txt","w") as f:
    model.summary(print_fn=lambda s: f.write(s+"\n"))

# ===============================================================
# 6. TRAINING
# ===============================================================
print("Starting Person3 fine-tuning...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    verbose=2
)

# ===============================================================
# 7. Save plots
# ===============================================================
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Person3 Accuracy")
plt.legend()
plt.savefig(f"{OUTPUT}/accuracy_p3.png")
plt.close()

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Person3 Loss")
plt.legend()
plt.savefig(f"{OUTPUT}/loss_p3.png")
plt.close()

# ===============================================================
# 8. Evaluate on test set
# ===============================================================
preds = model.predict(X_test).argmax(axis=1)

report = classification_report(y_test, preds)
with open(f"{OUTPUT}/p3_classification_report.txt","w") as f:
    f.write(report)

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Person3 Confusion Matrix")
plt.savefig(f"{OUTPUT}/confusion_matrix_p3.png")
plt.close()

# ===============================================================
# 9. Save final model
# ===============================================================
model.save(f"{OUTPUT}/mobilenetv2_person3_finetuned.h5")

print("Person3 fine-tuning complete.")
