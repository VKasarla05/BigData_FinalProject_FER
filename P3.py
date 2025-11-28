import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ================================
# PATHS (MATCHING YOUR VM)
# ================================
BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
OUTPUT = f"{BASE}/Output_3"
MODEL_FROM_P2 = f"{BASE}/Output_2/mobilenetv2_person2.h5"

os.makedirs(OUTPUT, exist_ok=True)

print("Saving all Person3 outputs to:", OUTPUT)

# ================================
# LOAD NPZ FILES
# ================================
train = np.load(f"{NPZ_PATH}/train.npz")
val   = np.load(f"{NPZ_PATH}/val.npz")
test  = np.load(f"{NPZ_PATH}/test.npz")

X_train, y_train = train["X"], train["y"]
X_val, y_val     = val["X"],   val["y"]
X_test, y_test   = test["X"],  test["y"]

print("\nLoaded:")
print("Train:", X_train.shape, "| Labels:", len(y_train))
print("Val:  ", X_val.shape,   "| Labels:", len(y_val))
print("Test: ", X_test.shape,  "| Labels:", len(y_test))

# MobileNetV2 expects 96×96×3
IMG_SIZE = 96

def preprocess(x):
    x = np.repeat(x[..., np.newaxis], 3, axis=-1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x

X_train = preprocess(X_train)
X_val   = preprocess(X_val)
X_test  = preprocess(X_test)

print("Final image shape:", X_train.shape)

# ================================
# LOAD PERSON2 MODEL
# ================================
print("\n🔄 Loading Person2 MobileNetV2 model...")
model = tf.keras.models.load_model(MODEL_FROM_P2)
print("Model loaded successfully!")

model.summary()

# ================================
# UNFREEZE TOP LAYERS (Light Finetuning)
# ================================
print("\n🔧 Unfreezing last 20 layers for Person3 fine-tuning...")
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ================================
# TRAIN FOR PERSON3
# ================================
print("\n🔥 Starting Person3 fine-tuning...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    verbose=2
)

# ================================
# PLOTS
# ================================
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend()
plt.title("Person3 Accuracy Curve")
plt.savefig(f"{OUTPUT}/accuracy_plot_p3.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.title("Person3 Loss Curve")
plt.savefig(f"{OUTPUT}/loss_plot_p3.png")
plt.close()

print("Saved accuracy & loss plots.")

# ================================
# EVALUATION ON TEST SET
# ================================
print("\n📊 Evaluating on Test Set...")
preds = model.predict(X_test).argmax(axis=1)

report = classification_report(y_test, preds)
print(report)

with open(f"{OUTPUT}/classification_report_p3.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Person3 Confusion Matrix")
plt.savefig(f"{OUTPUT}/confusion_matrix_p3.png")
plt.close()

# ================================
# SAVE FINAL PERSON3 MODEL
# ================================
FINAL_MODEL = f"{OUTPUT}/mobilenetv2_person3_finetuned.h5"
model.save(FINAL_MODEL)
print("💾 Saved Person3 final model to:", FINAL_MODEL)

print("\n🚀 PERSON 3 COMPLETE — Fine-tuning finished successfully!")
