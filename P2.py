import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==============================================================
# 1. MAKE OUTPUT FOLDER
# ==============================================================
output_dir = "/home/sat3812/Final_project/Output_2"
os.makedirs(output_dir, exist_ok=True)

print("Saving Person2 outputs to:", output_dir)

# ==============================================================
# 2. LOAD NPZ FILES FROM PERSON1
# ==============================================================
train = np.load("/home/sat3812/Final_project/Dataset/npz/train.npz")
val   = np.load("/home/sat3812/Final_project/Dataset/npz/val.npz")
test  = np.load("/home/sat3812/Final_project/Dataset/npz/test.npz")

X_train, y_train = train["X"], train["y"]
X_val, y_val     = val["X"],   val["y"]
X_test, y_test   = test["X"],  test["y"]

print("Loaded:")
print("Train:", X_train.shape, "| Labels:", len(y_train))
print("Val:  ", X_val.shape,   "| Labels:", len(y_val))
print("Test: ", X_test.shape,  "| Labels:", len(y_test))

# ==============================================================
# 3. PREPROCESSING — Resize + Convert to RGB
# ==============================================================
IMG_SIZE = 96  # MobileNet-friendly small size

def prep_images(X):
    X = np.repeat(X[..., np.newaxis], 3, axis=-1)  # grayscale → 3-channel
    X = tf.image.resize(X, (IMG_SIZE, IMG_SIZE)).numpy()
    return X

X_train = prep_images(X_train)
X_val   = prep_images(X_val)
X_test  = prep_images(X_test)

print("Final image shape:", X_train.shape)

# ==============================================================
# 4. MOBILENETV2 TRANSFER LEARNING
# ==============================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze backbone → FAST!!

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(7, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save model architecture
with open(os.path.join(output_dir, "mobilenet_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("\n🔥 Training MobileNetV2... (Very Fast)\n")

# ==============================================================
# 5. TRAIN MODEL
# ==============================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=64,
    verbose=2
)

# ==============================================================
# 6. SAVE TRAINING PLOTS
# ==============================================================
plt.figure(figsize=(7,5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("MobileNetV2 Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
plt.close()

plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("MobileNetV2 Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_plot.png"))
plt.close()

# ==============================================================
# 7. TEST EVALUATION
# ==============================================================
print("\n📊 Evaluating on Test Set...\n")
pred = model.predict(X_test).argmax(axis=1)

report = classification_report(y_test, pred)
cm = confusion_matrix(y_test, pred)

# Save classification report
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("MobileNetV2 Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

print(report)
print("\n🔥 PERSON 2 COMPLETE — MobileNetV2 Transfer Learning finished!")
