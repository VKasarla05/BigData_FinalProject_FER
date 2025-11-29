import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ===============================================================
# PATHS
# ===============================================================
BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
OUTPUT_DIR = f"{BASE}/Output_4"
MODEL_PATH = f"{BASE}/Output_P3/mobilenetv2finetuned.h5"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving Person4 outputs to:", OUTPUT_DIR)

# ===============================================================
# LOAD NPZ FILES
# ===============================================================
train = np.load(f"{NPZ_PATH}/train.npz")
val   = np.load(f"{NPZ_PATH}/val.npz")
test  = np.load(f"{NPZ_PATH}/test.npz")

X_test, y_test = test["X"], test["y"]

print("Test shape:", X_test.shape)

# ===============================================================
# PREPROCESS TEST DATA (match 96x96x3 used in Person2 & 3)
# ===============================================================
IMG_SIZE = 96

def preprocess(x):
    x = np.repeat(x[..., np.newaxis], 3, axis=-1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x

X_test = preprocess(X_test)
print("Final test shape:", X_test.shape)

# ===============================================================
# LOAD FINE-TUNED MODEL (Person3)
# ===============================================================
print("Loading Person3 fine-tuned model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ===============================================================
# EVALUATE MODEL
# ===============================================================
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

# ===============================================================
# PREDICTIONS
# ===============================================================
pred_probs = model.predict(X_test, verbose=0)
preds = np.argmax(pred_probs, axis=1)

# ===============================================================
# SAVE CLASSIFICATION REPORT
# ===============================================================
report = classification_report(y_test, preds)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)
print("Classification report saved.")

# ===============================================================
# CONFUSION MATRIX
# ===============================================================
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()
print("Confusion matrix saved.")

# ===============================================================
# FIND LAST CONV LAYER (bulletproof recursive version)
# ===============================================================
def get_last_conv_layer(model):
    # Recursively search inside nested models
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return get_last_conv_layer(layer)
            except:
                pass
        # Find any convolution layer
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No convolutional layer found for Grad-CAM.")

last_conv_name = get_last_conv_layer(model)
print("Last conv layer detected:", last_conv_name)

# ===============================================================
# GRAD-CAM FUNCTION
# ===============================================================
def make_gradcam_heatmap(img, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img, axis=0))
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ===============================================================
# GENERATE GRAD-CAM FOR EACH CLASS
# ===============================================================
classes = np.unique(y_test)
print("Generating Grad-CAM heatmaps...")

for cls in classes:
    idx = np.where(y_test == cls)[0][0]  # take first sample of class
    img = X_test[idx]
    heatmap = make_gradcam_heatmap(img, model, last_conv_name)

    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap="jet")
    plt.title(f"Grad-CAM Class {cls}")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/gradcam_class_{cls}.png")
    plt.close()

print("Grad-CAM heatmaps saved.")

# ===============================================================
# SAVE MISCLASSIFIED SAMPLES
# ===============================================================
misclassified = np.where(preds != y_test)[0][:10]
print("Saving misclassified samples...")

for i, idx in enumerate(misclassified):
    img = X_test[idx]

    plt.figure(figsize=(3, 3))
    plt.imshow(img.astype("uint8"))
    plt.title(f"True: {y_test[idx]} | Pred: {preds[idx]}")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/misclassified_{i}.png")
    plt.close()

print("Misclassified sample images saved.")
print("Person4 processing completed.")
