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
OUTPUT = f"{BASE}/Output_4"
MODEL_P3 = f"{BASE}/Output_P3/mobilenetv2finetuned.h5"

os.makedirs(OUTPUT, exist_ok=True)
print("Saving Person4 outputs to:", OUTPUT)

# ===============================================================
# LOAD NPZ FILES
# ===============================================================
train_npz = np.load(f"{NPZ_PATH}/train.npz")
val_npz   = np.load(f"{NPZ_PATH}/val.npz")
test_npz  = np.load(f"{NPZ_PATH}/test.npz")

X_test, y_test = test_npz["X"], test_npz["y"]

print("Loaded dataset:")
print("Test:", X_test.shape, "| Labels:", len(y_test))

# ===============================================================
# PREPROCESS TEST DATA
# ===============================================================
IMG_SIZE = 96

def preprocess(x):
    x = np.repeat(x[..., np.newaxis], 3, axis=-1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x

X_test = preprocess(X_test)
print("Final test shape:", X_test.shape)

# ===============================================================
# LOAD FINE-TUNED MODEL (PERSON3)
# ===============================================================
print("Loading Person3 fine-tuned model...")
model = tf.keras.models.load_model(MODEL_P3)
print("Model loaded successfully.")

# ===============================================================
# EVALUATE ON TEST SET
# ===============================================================
print("Evaluating on test set...")
preds = model.predict(X_test, verbose=2)
y_pred = preds.argmax(axis=1)

test_acc = np.mean(y_pred == y_test)

print("Test accuracy:", test_acc)
print("Test loss:", tf.keras.losses.sparse_categorical_crossentropy(y_test, preds).numpy().mean())

# Save classification report
report = classification_report(y_test, y_pred)
with open(f"{OUTPUT}/person4_classification_report.txt", "w") as f:
    f.write(report)

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/confusion_matrix.png")
plt.close()

print("Classification report and confusion matrix saved.")

# ===============================================================
# BUILD GRAD-CAM MODEL
# ===============================================================

# Last conv layer in MobileNetV2
LAST_CONV = "block_16_project"   # Verified from your model summary
print("Last conv layer detected:", LAST_CONV)

last_conv_layer = model.get_layer(LAST_CONV)

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)

# ===============================================================
# GRAD-CAM FUNCTION (ERROR-FREE)
# ===============================================================
def generate_gradcam(img, true_label, idx):
    img_tensor = tf.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        pred_class = tf.argmax(preds[0])
        class_channel = preds[:, pred_class]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap="jet")
    plt.title(f"Grad-CAM sample {idx}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/gradcam_class_{idx}.png")
    plt.close()

# ===============================================================
# GENERATE GRAD-CAM FOR 7 RANDOM CORRECTLY CLASSIFIED SAMPLES
# ===============================================================
print("Generating Grad-CAM heatmaps...")

correct_indices = np.where(y_pred == y_test)[0]
chosen = np.random.choice(correct_indices, 7, replace=False)

for i in chosen:
    generate_gradcam(X_test[i], y_test[i], i)

print("Grad-CAM heatmaps saved.")

# ===============================================================
# SAVE 10 MISCLASSIFIED IMAGES
# ===============================================================
print("Saving misclassified samples...")

incorrect_indices = np.where(y_pred != y_test)[0][:10]

for j, idx in enumerate(incorrect_indices):
    plt.figure(figsize=(4, 4))
    plt.imshow(X_test[idx] / 255.0)
    plt.title(f"True={y_test[idx]}  Pred={y_pred[idx]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/misclassified_{j}.png")
    plt.close()

print("Misclassified sample images saved.")
print("Person4 processing completed.")
