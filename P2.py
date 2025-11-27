# =====================================================================
# PERSON 2 — Transfer Learning with ResNet50 (Using Person1 NPZ files)
# Saves all outputs to: /home/sat3812/Final_project/Output_2
# =====================================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# =====================================================================
# 1. PATHS + OUTPUT FOLDER
# =====================================================================
npz_path = "/home/sat3812/Final_project/Dataset/npz"
output_dir = "/home/sat3812/Final_project/Output_2"
os.makedirs(output_dir, exist_ok=True)

print("Using NPZ files from:", npz_path)
print("Saving all Person2 outputs to:", output_dir)

# =====================================================================
# 2. LOAD DATA FROM PERSON 1
# =====================================================================
train = np.load(os.path.join(npz_path, "train.npz"))
val   = np.load(os.path.join(npz_path, "val.npz"))
test  = np.load(os.path.join(npz_path, "test.npz"))

X_train, y_train = train["X"], train["y"]
X_val, y_val     = val["X"],   val["y"]
X_test, y_test   = test["X"],  test["y"]

print("Loaded:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# =====================================================================
# 3. RESHAPE + CONVERT TO RGB (ResNet requires 3 channels)
# =====================================================================
IMG_SIZE = 96     # Smaller = faster CPU inference
NUM_CLASSES = 7   # 7 emotion classes

# Expand grayscale → RGB
X_train = np.repeat(X_train[..., np.newaxis], 3, axis=-1)
X_val   = np.repeat(X_val[..., np.newaxis],   3, axis=-1)
X_test  = np.repeat(X_test[..., np.newaxis],  3, axis=-1)

# Resize images to 96×96
X_train = tf.image.resize(X_train, (IMG_SIZE, IMG_SIZE)).numpy()
X_val   = tf.image.resize(X_val,   (IMG_SIZE, IMG_SIZE)).numpy()
X_test  = tf.image.resize(X_test,  (IMG_SIZE, IMG_SIZE)).numpy()

# =====================================================================
# 4. LIGHT DATA AUGMENTATION (Improves accuracy significantly)
# =====================================================================
data_augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.15),
], name="data_augmentation")

# =====================================================================
# 5. BUILD RESNET50 TRANSFER LEARNING MODEL
# =====================================================================
base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False    # Freeze full backbone (CPU SAFE)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmenter(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save model summary
with open(os.path.join(output_dir, "resnet_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("✔ Saved: resnet_model_summary.txt")

# =====================================================================
# 6. TRAIN RESNET MODEL
# =====================================================================
print("\n🔥 Training ResNet50 Transfer Learning Model...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,            # Perfect for CPU; lightweight
    batch_size=32,
    verbose=2
)

# =====================================================================
# 7. SAVE TRAINING CURVE PLOTS
# =====================================================================
def save_plot(history, key, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(history.history[key], label="Train")
    plt.plot(history.history["val_" + key], label="Validation")
    plt.title(f"ResNet TL {key.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

save_plot(history, "accuracy", "resnet_accuracy.png")
save_plot(history, "loss", "resnet_loss.png")

print("✔ Saved accuracy & loss plots")

# =====================================================================
# 8. EVALUATE RESNET ON TEST SET
# =====================================================================
pred_probs = model.predict(X_test, verbose=0)
pred_labels = pred_probs.argmax(axis=1)

# Classification report
report = classification_report(y_test, pred_labels)
with open(os.path.join(output_dir, "resnet_classification_report.txt"), "w") as f:
    f.write(report)

print("\n📄 Classification Report Saved!")

# =====================================================================
# 9. CONFUSION MATRIX
# =====================================================================
cm = confusion_matrix(y_test, pred_labels)
np.save(os.path.join(output_dir, "resnet_confusion_matrix.npy"), cm)

plt.figure(figsize=(7,7))
plt.imshow(cm, cmap="Blues")
plt.title("ResNet Confusion Matrix")
plt.colorbar()
plt.savefig(os.path.join(output_dir, "resnet_confusion_matrix.png"))
plt.close()

print("✔ Confusion matrix saved")

# =====================================================================
# 10. PREDICTION DISTRIBUTION PLOT
# =====================================================================
plt.figure(figsize=(8,4))
plt.hist(pred_labels, bins=NUM_CLASSES, color="orange")
plt.title("Prediction Distribution")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "prediction_distribution.png"))
plt.close()

print("✔ Prediction distribution plot saved")

print("\n🎯 PERSON 2 COMPLETED — All files saved to:", output_dir)
