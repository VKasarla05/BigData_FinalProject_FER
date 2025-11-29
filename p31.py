import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession

# =====================================================
# 1. SPARK SESSION (Required by your professor)
# =====================================================
spark = SparkSession.builder \
    .appName("Person2-ResNet18") \
    .master("spark://192.168.13.134:7077") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark session started.")

# =====================================================
# 2. PATHS
# =====================================================
BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
OUTPUT = f"{BASE}/Output_2"

os.makedirs(OUTPUT, exist_ok=True)
print("Saving Person2 outputs to:", OUTPUT)

# =====================================================
# 3. LOAD NPZ USING SPARK
# =====================================================
train_npz = np.load(f"{NPZ_PATH}/train.npz")
val_npz   = np.load(f"{NPZ_PATH}/val.npz")
test_npz  = np.load(f"{NPZ_PATH}/test.npz")

X_train, y_train = train_npz["X"], train_npz["y"]
X_val, y_val     = val_npz["X"],   val_npz["y"]
X_test, y_test   = test_npz["X"],  test_npz["y"]

print("Loaded datasets:")
print("Train:", X_train.shape)
print("Val:",   X_val.shape)
print("Test:",  X_test.shape)

# Stop Spark after loading dataset
spark.stop()
print("Spark session stopped.")

# =====================================================
# 4. PREPROCESSING FOR RESNET
# =====================================================
IMG_SIZE = 96

def preprocess(x):
    x = np.repeat(x[..., np.newaxis], 3, axis=-1)       # grayscale → RGB
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x.astype("float32") / 255.0

X_train = preprocess(X_train)
X_val   = preprocess(X_val)
X_test  = preprocess(X_test)

# =====================================================
# 5. BUILD RESNET18 (MANUAL KERAS VERSION)
# =====================================================
def conv_bn_relu(x, filters, kernel, stride=1):
    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride,
                               padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1
    shortcut = x

    x = conv_bn_relu(x, filters, 3, stride)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if downsample:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=2,
                                          use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.ReLU()(x)

def build_resnet18():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = conv_bn_relu(inputs, 64, 3)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)

    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(7, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model

model = build_resnet18()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save model summary
with open(os.path.join(OUTPUT, "resnet18_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# =====================================================
# 6. TRAIN MODEL
# =====================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=64,
    verbose=2
)

# =====================================================
# 7. SAVE TRAINING PLOTS
# =====================================================
plt.figure()
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("ResNet18 Accuracy")
plt.legend()
plt.savefig(os.path.join(OUTPUT, "accuracy_plot.png"))
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("ResNet18 Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT, "loss_plot.png"))
plt.close()

# =====================================================
# 8. EVALUATE MODEL
# =====================================================
preds = model.predict(X_test).argmax(axis=1)

report = classification_report(y_test, preds)
with open(os.path.join(OUTPUT, "classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("ResNet18 Confusion Matrix")
plt.savefig(os.path.join(OUTPUT, "confusion_matrix.png"))
plt.close()

# =====================================================
# 9. SAVE FINAL MODEL
# =====================================================
model.save(f"{OUTPUT}/resnet18_person2.h5")
print("Saved Person2 ResNet18 model.")
