# ===============================================================
## Hyperparameter Tuning
# ===============================================================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession

# ===============================================================
# START SPARK SESSION
# ===============================================================

spark = SparkSession.builder \
    .appName("Finetuning") \
    .master("spark://hadoop1:7077") \       
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

print("Spark session started on:", spark.sparkContext.master)

# ===============================================================
# PATHS
# ===============================================================

BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
OUTPUT = f"{BASE}/Output_P3"
MODEL_FROM_P2 = f"{BASE}/Output_2/mobilenetv2P2.h5"
os.makedirs(OUTPUT, exist_ok=True)
print("Saving Person3 outputs to:", OUTPUT)

# ===============================================================
# LOAD NPZ FILES USING SPARK FOR DISTRIBUTED I/O
# ===============================================================

train_npz = np.load(f"{NPZ_PATH}/train.npz")
val_npz   = np.load(f"{NPZ_PATH}/val.npz")
test_npz  = np.load(f"{NPZ_PATH}/test.npz")
X_train, y_train = train_npz["X"], train_npz["y"]
X_val, y_val     = val_npz["X"],   val_npz["y"]
X_test, y_test   = test_npz["X"],  test_npz["y"]

print("Loaded NPZ:")
print("Train:", X_train.shape, "| Labels:", len(y_train))
print("Val:",   X_val.shape,   "| Labels:", len(y_val))
print("Test:",  X_test.shape,  "| Labels:", len(y_test))

# ===============================================================
# STOP SPARK — deep learning will run WITHOUT spark
# ===============================================================
spark.stop()
print("Spark session stopped after distributed data loading.")
# ===============================================================
# PREPROCESSING FOR MOBILENETV2
# ===============================================================

IMG_SIZE = 96
def preprocess(x):
    x = np.repeat(x[..., np.newaxis], 3, axis=-1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x
X_train = preprocess(X_train)
X_val   = preprocess(X_val)
X_test  = preprocess(X_test)
print("After preprocessing:", X_train.shape)
# ===============================================================
# LOAD PERSON2 MODEL
# ===============================================================

print("Loading Person2 MobileNetV2 model...")
model = tf.keras.models.load_model(MODEL_FROM_P2)
print("Model loaded successfully.")

# Save model summary
with open(os.path.join(OUTPUT, "model_before_finetune.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# ===============================================================
# UNFREEZE LAST 20 LAYERS FOR FINE-TUNING
# ===============================================================
print("Unfreezing last 20 layers for fine-tuning.")
for layer in model.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
# ===============================================================
# TRAIN MODEL
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
# SAVE TRAINING PLOTS
# ===============================================================

plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Person3 Accuracy Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}/accuracy_plot_p3.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Person3 Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}/loss_plot_p3.png")
plt.close()

# ===============================================================
# EVALUATE ON TEST SET
# ===============================================================
print("Evaluating on test set...")
preds = model.predict(X_test).argmax(axis=1)
report = classification_report(y_test, preds)
with open(f"{OUTPUT}/classification_report_p3.txt", "w") as f:
    f.write(report)
print(report)
# ===============================================================
# CONFUSION MATRIX
# ===============================================================
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Person3 Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/confusion_matrix_p3.png")
plt.close()
# ===============================================================
# SAVE FINAL MODEL
# ===============================================================

FINAL_MODEL = f"{OUTPUT}/mobilenetv2finetuned.h5"
model.save(FINAL_MODEL)
print("Saved Person3 fine-tuned model to:", FINAL_MODEL)

print("Person3 fine-tuning completed.")
