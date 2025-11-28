# ===============================================================
# PERSON 4 — Model Evaluation + Interpretability (Grad-CAM)
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
NPZ_PATH = "/home/sat3812/Final_project/Dataset/npz"
MODEL_PATH = "/home/sat3812/Final_project/Output_3/mobilenetv2_person3_finetuned.h5"
OUTPUT_DIR = "/home/sat3812/Final_project/Output_4"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving Person4 outputs to:", OUTPUT_DIR)


# ===============================================================
# 2. LOAD NPZ FILES
# ===============================================================
train = np.load(f"{NPZ_PATH}/train.npz")
val = np.load(f"{NPZ_PATH}/val.npz")
test = np.load(f"{NPZ_PATH}/test.npz")

X_test, y_test = test["X"], test["y"]

print("Loaded NPZ files")
print("Test:", X_test.shape, "| Labels:", y_test.shape)

# Resize images for MobileNetV2 (96x96 RGB)
X_test = np.repeat(X_test[..., np.newaxis], 3, axis=-1)
X_test = tf.image.resize(X_test, (96, 96)).numpy()


# ===============================================================
# 3. LOAD TRAINED MODEL (Person3)
# ===============================================================
print("\nLoading Person3 model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


# ===============================================================
# 4. EVALUATE MODEL
# ===============================================================
print("\nEvaluating on Test Set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

# Predictions
preds = model.predict(X_test).argmax(axis=1)

# Save classification report
report = classification_report(y_test, preds)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

print("\nClassification report saved.")


# ===============================================================
# 5. CONFUSION MATRIX
# ===============================================================
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()
print("Confusion matrix saved.")


# ===============================================================
# 6. AUTO-FIND LAST CONV LAYER FOR GRAD-CAM
# ===============================================================
def find_last_conv_layer(model):
    """Automatically detect last Conv2D layer in functional/nested models."""
    # Search top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"✔ Last Conv layer found: {layer.name}")
            return layer.name

    # Search nested layers (MobileNetV2 case)
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    print(f"✔ Last Conv layer found inside nested block: {sub.name}")
                    return sub.name

    raise ValueError("❌ No Conv2D layer found for Grad-CAM.")


last_conv_layer_name = find_last_conv_layer(model)

# Build Grad-CAM model
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)


# ===============================================================
# 7. GRAD-CAM FUNCTION
# ===============================================================
def generate_gradcam(image_array, label, save_path):
    """Generate & save Grad-CAM heatmap."""
    img_tensor = tf.expand_dims(image_array, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    # Resize heatmap
    heatmap = tf.image.resize(cam[..., np.newaxis], (96, 96)).numpy().squeeze()

    # Plot
    plt.figure(figsize=(4, 4))
    plt.imshow(image_array.astype("uint8"))
    plt.imshow(heatmap, cmap='jet', alpha=0.45)
    plt.title(f"Grad-CAM (Pred: {label})")
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


# ===============================================================
# 8. GENERATE GRAD-CAM FOR SAMPLE IMAGES (ALL CLASSES)
# ===============================================================
print("\nGenerating Grad-CAM heatmaps...")

unique_labels = np.unique(y_test)
for cls in unique_labels:
    idx = np.where(y_test == cls)[0][0]  # pick first instance of each class
    img = (X_test[idx] * 255).numpy().astype("uint8")
    pred = preds[idx]

    save_path = f"{OUTPUT_DIR}/gradcam_class_{cls}.png"
    generate_gradcam(img, pred, save_path)

print("Grad-CAM heatmaps saved.")


# ===============================================================
# 9. SAVE MISCLASSIFIED SAMPLES
# ===============================================================
print("\nSaving misclassified samples...")
mis_idx = np.where(preds != y_test)[0][:10]   # first 10 mistakes

for i, idx in enumerate(mis_idx):
    img = (X_test[idx] * 255).numpy().astype("uint8")
    true_label = y_test[idx]
    pred_label = preds[idx]

    plt.figure(figsize=(3, 3))
    plt.imshow(img.astype("uint8"))
    plt.title(f"True: {true_label} | Pred: {pred_label}")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/misclassified_{i}.png")
    plt.close()

print("Misclassified image samples saved.")

# ===============================================================
print("\n🎉 PERSON 4 COMPLETE — All evaluation outputs generated successfully!")
# ===============================================================
