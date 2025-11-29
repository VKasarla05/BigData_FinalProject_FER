import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ======================================================
# Paths
# ======================================================
BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
OUTPUT_DIR = f"{BASE}/Output_4"
MODEL_PATH = f"{BASE}/Output_P3/mobilenetv2finetuned.h5"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving Person4 outputs to:", OUTPUT_DIR)

# ======================================================
# Load NPZ files
# ======================================================
train = np.load(f"{NPZ_PATH}/train.npz")
val   = np.load(f"{NPZ_PATH}/val.npz")
test  = np.load(f"{NPZ_PATH}/test.npz")

X_test, y_test = test["X"], test["y"]
X_test = X_test[..., np.newaxis]

print("Test shape:", X_test.shape)

# ======================================================
# Load Person3 model
# ======================================================
print("Loading Person3 fine-tuned model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ======================================================
# Preprocess test images to MobileNet size (96×96×3)
# ======================================================
IMG_SIZE = 96

def prep(x):
    x = np.repeat(x, 3, axis=-1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x

X_test_resized = prep(X_test)

print("Final test shape:", X_test_resized.shape)

# ======================================================
# Evaluate model on test set
# ======================================================
print("Evaluating on test set...")
pred_probs = model.predict(X_test_resized, verbose=0)
preds = np.argmax(pred_probs, axis=1)

report = classification_report(y_test, preds)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

print(report)

# ======================================================
# Confusion Matrix
# ======================================================
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# ======================================================
# Identify last convolution layer safely
# ======================================================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No convolution layer found.")

last_conv_name = get_last_conv_layer(model)
print("Last Conv Layer:", last_conv_name)

# ======================================================
# Grad-CAM generator
# ======================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(np.expand_dims(img_array, axis=0))
        class_index = tf.argmax(preds[0])
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# ======================================================
# Generate Grad-CAM for 1 sample per class
# ======================================================
classes = np.unique(y_test)

for cls in classes:
    idx = np.where(y_test == cls)[0][0]
    img = X_test_resized[idx]

    heatmap = make_gradcam_heatmap(img, model, last_conv_name)

    plt.figure(figsize=(4,4))
    plt.imshow(heatmap, cmap="jet")
    plt.title(f"Grad-CAM Class {cls}")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/gradcam_class_{cls}.png")
    plt.close()

# ======================================================
# Save misclassified samples
# ======================================================
mis_idx = np.where(preds != y_test)[0][:10]

for i, idx in enumerate(mis_idx):
    img = X_test[idx].squeeze()

    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap="gray")
    plt.title(f"True: {y_test[idx]}  Pred: {preds[idx]}")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/misclassified_{i}.png")
    plt.close()

print("Person4 completed. All outputs saved.")
