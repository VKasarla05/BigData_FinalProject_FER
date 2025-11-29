# ===============================================================
# PERSON 4 — Model Evaluation + Grad-CAM + Error Analysis
# ===============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ===============================================================
# 1. PATHS
# ===============================================================
BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
MODEL_PATH = f"{BASE}/Output_P3/mobilenetv2finetuned.h5"
OUTPUT_DIR = f"{BASE}/Output_4"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving Person4 outputs to:", OUTPUT_DIR)

# ===============================================================
# 2. LOAD NPZ FILES
# ===============================================================
train = np.load(f"{NPZ_PATH}/train.npz")
val   = np.load(f"{NPZ_PATH}/val.npz")
test  = np.load(f"{NPZ_PATH}/test.npz")

X_test, y_test = test["X"], test["y"]

print("Loaded NPZ files")
print("Test:", X_test.shape, "| Labels:", len(y_test))

# Add channel dimension
X_test = X_test[..., np.newaxis]

# ===============================================================
# 3. PREPROCESS FOR MOBILENETV2
# ===============================================================
IMG_SIZE = 96

def preprocess(x):
    x = np.repeat(x, 3, axis=-1)  # grayscale → RGB
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x

X_test_rgb = preprocess(X_test)

print("Final test shape:", X_test_rgb.shape)

# ===============================================================
# 4. LOAD PERSON3 MODEL
# ===============================================================
print("Loading Person3 fine-tuned model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ===============================================================
# 5. EVALUATE MODEL
# ===============================================================
print("Evaluating on test set...")
test_preds = model.predict(X_test_rgb, verbose=1).argmax(axis=1)

report = classification_report(y_test, test_preds)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

print(report)

# ===============================================================
# 6. CONFUSION MATRIX
# ===============================================================
cm = confusion_matrix(y_test, test_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

print("Confusion matrix saved.")

# ===============================================================
# 7. GRAD-CAM IMPLEMENTATION
# ===============================================================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("No conv layer found.")

last_conv_name = get_last_conv_layer(model)
print("Last conv layer for Grad-CAM:", last_conv_name)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0,1))

    conv_output = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

# ===============================================================
# 8. SAVE GRAD-CAM HEATMAPS FOR FIRST SAMPLE OF EACH CLASS
# ===============================================================
unique_classes = np.unique(y_test)

for c in unique_classes:
    idx = np.where(y_test == c)[0][0]
    img = X_test_rgb[idx:idx+1]

    heatmap = make_gradcam_heatmap(img, model, last_conv_name)

    plt.figure(figsize=(4,4))
    plt.imshow(heatmap, cmap="jet")
    plt.title(f"Grad-CAM Class {c}")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/gradcam_class_{c}.png")
    plt.close()

print("Grad-CAM heatmaps saved.")

# ===============================================================
# 9. SAVE MISCLASSIFIED SAMPLES
# ===============================================================
mis_idx = np.where(test_preds != y_test)[0][:10]

for i, idx in enumerate(mis_idx):
    img = X_test[idx].squeeze()

    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap="gray")
    plt.title(f"True {y_test[idx]} | Pred {test_preds[idx]}")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/misclassified_{i}.png")
    plt.close()

print("Misclassified samples saved.")
print("Person4 tasks complete.")
