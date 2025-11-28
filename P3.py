import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================================
# 1. OUTPUT DIRECTORY
# ==============================================================
output_dir = "/home/sat3812/Final_project/Output_3"
os.makedirs(output_dir, exist_ok=True)
print("Saving Person3 outputs to:", output_dir)

# ==============================================================
# 2. LOAD SAME NPZ FILES FROM PERSON1
# ==============================================================
train = np.load("/home/sat3812/Final_project/Dataset/npz/train.npz")
val   = np.load("/home/sat3812/Final_project/Dataset/npz/val.npz")
test  = np.load("/home/sat3812/Final_project/Dataset/npz/test.npz")

X_train, y_train = train["X"], train["y"]
X_val, y_val     = val["X"],   val["y"]
X_test, y_test   = test["X"],  test["y"]

print("Loaded:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# ==============================================================
# 3. PREPROCESSING — RESIZE + 3-CHANNEL RGB
# ==============================================================
IMG_SIZE = 96

def prep(X):
    X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    return tf.image.resize(X, (IMG_SIZE, IMG_SIZE)).numpy()

X_train = prep(X_train)
X_val   = prep(X_val)
X_test  = prep(X_test)

# ==============================================================
# 4. DATA AUGMENTATION (STRONGER THAN PERSON 2)
# ==============================================================
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2)
])

# ==============================================================
# 5. LOAD MOBILENETV2 BASE MODEL
#    BUT NOW WE WILL FINE-TUNE LAST 25 LAYERS
# ==============================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# 🔓 Unfreeze top 25 layers for fine-tuning
for layer in base_model.layers[:-25]:
    layer.trainable = False
for layer in base_model.layers[-25:]:
    layer.trainable = True

# ==============================================================
# 6. BUILD FINE-TUNING MODEL
# ==============================================================

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_aug(inputs)  # apply augmentation
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# TRY MULTIPLE UNITS (PERSON-3 CONTRIBUTION)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)

outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# SMALL LR for Fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save summary
with open(os.path.join(output_dir, "P3_finetuned_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# ==============================================================
# 7. TRAIN — Fine-tuning (slow but effective)
# ==============================================================
print("\n🔥 Training Person3 Fine-Tuning Model...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=2
)

# ==============================================================
# 8. SAVE ACCURACY + LOSS PLOTS
# ==============================================================
plt.figure(figsize=(7,5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("P3 Fine-Tuned Model Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, "accuracy_P3.png"))
plt.close()

plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("P3 Fine-Tuned Model Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_P3.png"))
plt.close()

# ==============================================================
# 9. TEST EVALUATION
# ==============================================================
pred = model.predict(X_test).argmax(axis=1)

report = classification_report(y_test, pred)
cm = confusion_matrix(y_test, pred)

with open(os.path.join(output_dir, "classification_report_P3.txt"), "w") as f:
    f.write(report)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix — Person3")
plt.savefig(os.path.join(output_dir, "cm_P3.png"))
plt.close()

print(report)
print("\n🎉 PERSON 3 COMPLETE — Fine-Tuned Model Saved!\n")
